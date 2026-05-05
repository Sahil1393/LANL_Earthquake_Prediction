import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "final_model.pkl"
    )

    model_report_file_path: str = os.path.join(
        "artifacts",
        "model_report.csv"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_models_and_params(self, fast_mode=True):
        """
        Returns models and hyperparameter search spaces.

        fast_mode=True is recommended for your laptop and partial-large data.
        """

        models = {
            "Ridge": Ridge(),

            "ElasticNet": ElasticNet(random_state=42),

            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                random_state=42,
                loss="absolute_error"
            ),

            "Extra Trees": ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1
            ),
        }

        params = {
            "Ridge": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
            },

            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
            },

            "Hist Gradient Boosting": {
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_iter": [100, 200, 300],
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": [0.0, 0.01, 0.1, 1.0]
            },

            "Extra Trees": {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
        }

        if not fast_mode:
            models["Random Forest"] = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

            params["Random Forest"] = {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                objective="reg:absoluteerror",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mae"
            )

            params["XGBoost"] = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "reg_lambda": [1.0, 3.0, 5.0]
            }

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                objective="mae",
                metric="mae",
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

            params["LightGBM"] = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 8, 12],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "reg_lambda": [0.0, 1.0, 3.0]
            }

        if CATBOOST_AVAILABLE:
            os.makedirs("catboost_info", exist_ok=True)

            models["CatBoost"] = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                random_seed=42,
                verbose=False,
                train_dir="catboost_info",
                allow_writing_files=False
            )

            params["CatBoost"] = {
                "iterations": [200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5, 7]
            }

        return models, params

    def get_cv_strategy(self, n_splits=5):
        """
        LANL data is time-ordered, so TimeSeriesSplit is better than random KFold.
        """

        return TimeSeriesSplit(n_splits=n_splits)

    def needs_scaling(self, model_name):
        """
        Tree models do not need scaling.
        Linear models benefit from scaling.
        """

        scaling_models = ["Ridge", "ElasticNet"]
        return model_name in scaling_models

    def clean_data(self, X, y):
        """
        Cleans feature and target data before training.
        """

        X = X.copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        y = np.asarray(y, dtype=np.float32)

        valid_mask = ~np.isnan(y)

        X = X.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]

        return X, y

    def train_single_model(
        self,
        model_name,
        model,
        params,
        X_train,
        y_train,
        n_iter=8
    ):
        """
        Trains one model using RandomizedSearchCV.
        """

        try:
            model_params = params.get(model_name, {})

            if len(model_params) == 0:
                model.fit(X_train, y_train)
                return model, {}

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=model_params,
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=3,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            return search.best_estimator_, search.best_params_

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_all_models(
        self,
        X,
        y,
        models,
        params,
        n_splits=5,
        n_iter=8
    ):
        """
        Evaluates all models using TimeSeriesSplit.
        """

        try:
            cv = self.get_cv_strategy(n_splits=n_splits)

            model_report = {}
            trained_model_store = {}

            for model_name, model in models.items():
                logging.info("=" * 80)
                logging.info(f"Starting training for model: {model_name}")

                fold_models = []
                fold_scalers = []
                fold_mae_scores = []
                fold_r2_scores = []
                fold_best_params = []

                for fold, (train_index, valid_index) in enumerate(cv.split(X)):
                    logging.info(f"{model_name} | Fold {fold + 1} started")

                    X_train = X.iloc[train_index]
                    X_valid = X.iloc[valid_index]

                    y_train = y[train_index]
                    y_valid = y[valid_index]

                    scaler = None

                    if self.needs_scaling(model_name):
                        scaler = StandardScaler()
                        X_train_model = scaler.fit_transform(X_train)
                        X_valid_model = scaler.transform(X_valid)
                    else:
                        X_train_model = X_train
                        X_valid_model = X_valid

                    trained_model, best_params = self.train_single_model(
                        model_name=model_name,
                        model=model,
                        params=params,
                        X_train=X_train_model,
                        y_train=y_train,
                        n_iter=n_iter
                    )

                    y_pred = trained_model.predict(X_valid_model)

                    # LANL time_to_failure target usually stays positive
                    y_pred = np.clip(y_pred, 0, None)

                    mae = mean_absolute_error(y_valid, y_pred)
                    r2 = r2_score(y_valid, y_pred)

                    fold_models.append(trained_model)
                    fold_scalers.append(scaler)
                    fold_mae_scores.append(float(mae))
                    fold_r2_scores.append(float(r2))
                    fold_best_params.append(best_params)

                    logging.info(
                        f"{model_name} | Fold {fold + 1} MAE: {mae:.6f}"
                    )
                    logging.info(
                        f"{model_name} | Fold {fold + 1} R2: {r2:.6f}"
                    )

                avg_mae = float(np.mean(fold_mae_scores))
                avg_r2 = float(np.mean(fold_r2_scores))
                std_mae = float(np.std(fold_mae_scores))

                model_report[model_name] = {
                    "avg_mae": avg_mae,
                    "std_mae": std_mae,
                    "avg_r2": avg_r2,
                    "fold_mae_scores": fold_mae_scores,
                    "fold_r2_scores": fold_r2_scores,
                }

                trained_model_store[model_name] = {
                    "models": fold_models,
                    "scalers": fold_scalers,
                    "best_params": fold_best_params,
                    "avg_mae": avg_mae,
                    "std_mae": std_mae,
                    "avg_r2": avg_r2,
                }

                logging.info(
                    f"{model_name} completed | "
                    f"Avg MAE: {avg_mae:.6f} | "
                    f"Std MAE: {std_mae:.6f} | "
                    f"Avg R2: {avg_r2:.6f}"
                )

            return model_report, trained_model_store

        except Exception as e:
            raise CustomException(e, sys)

    def save_model_report(self, model_report):
        """
        Saves model comparison report as CSV.
        """

        rows = []

        for model_name, report in model_report.items():
            rows.append(
                {
                    "model_name": model_name,
                    "avg_mae": report["avg_mae"],
                    "std_mae": report["std_mae"],
                    "avg_r2": report["avg_r2"],
                    "fold_mae_scores": report["fold_mae_scores"],
                    "fold_r2_scores": report["fold_r2_scores"],
                }
            )

        report_df = pd.DataFrame(rows)
        report_df = report_df.sort_values(by="avg_mae", ascending=True)

        report_df.to_csv(
            self.model_trainer_config.model_report_file_path,
            index=False
        )

        logging.info(
            f"Model report saved at: "
            f"{self.model_trainer_config.model_report_file_path}"
        )

    def initiate_model_trainer(
        self,
        X,
        y,
        fast_mode=True,
        n_splits=5,
        n_iter=8
    ):
        """
        Best model trainer for your LANL project.

        Uses:
        - TimeSeriesSplit
        - MAE for model selection
        - LightGBM / XGBoost / CatBoost if installed
        - HistGradientBoosting and ExtraTrees fallback
        """

        try:
            logging.info("Entered model trainer component")
            logging.info(f"Fast mode: {fast_mode}")
            logging.info(f"CV splits: {n_splits}")
            logging.info(f"Random search iterations: {n_iter}")

            X, y = self.clean_data(X, y)

            if len(X) < 50:
                raise ValueError(
                    f"Not enough training samples: {len(X)}. "
                    "Increase ingestion max_chunks or reduce step_size."
                )

            models, params = self.get_models_and_params(fast_mode=fast_mode)

            model_report, trained_model_store = self.evaluate_all_models(
                X=X,
                y=y,
                models=models,
                params=params,
                n_splits=n_splits,
                n_iter=n_iter
            )

            best_model_name = min(
                model_report,
                key=lambda name: model_report[name]["avg_mae"]
            )

            best_model_mae = model_report[best_model_name]["avg_mae"]
            best_model_r2 = model_report[best_model_name]["avg_r2"]
            best_model_std_mae = model_report[best_model_name]["std_mae"]

            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Best model MAE: {best_model_mae}")
            logging.info(f"Best model MAE std: {best_model_std_mae}")
            logging.info(f"Best model R2: {best_model_r2}")

            self.save_model_report(model_report)

            best_model_object = trained_model_store[best_model_name]

            final_model_object = {
                "best_model_name": best_model_name,
                "best_model_mae": best_model_mae,
                "best_model_std_mae": best_model_std_mae,
                "best_model_r2": best_model_r2,
                "models": best_model_object["models"],
                "scalers": best_model_object["scalers"],
                "best_params": best_model_object["best_params"],
                "feature_columns": list(X.columns),
                "model_report": model_report,
                "all_model_results": trained_model_store,
                "target_clip_min": 0,
            }

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model_object
            )

            logging.info(
                f"Final model saved at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            return {
                "best_model_name": best_model_name,
                "best_model_mae": best_model_mae,
                "best_model_std_mae": best_model_std_mae,
                "best_model_r2": best_model_r2,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "model_report_path": self.model_trainer_config.model_report_file_path,
                "number_of_models": len(best_model_object["models"]),
                "training_samples": len(X),
                "number_of_features": X.shape[1],
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("ModelTrainer is used by train_pipeline.py.")
    print("Run: python -m src.pipeline.train_pipeline")