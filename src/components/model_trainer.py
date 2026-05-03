import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_models_and_params(self):
        models = {
            "Linear Regression": LinearRegression(),

            "Ridge": Ridge(),

            "ElasticNet": ElasticNet(),

            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),

            "Extra Trees": ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),

            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                random_state=42
            ),

            "AdaBoost": AdaBoostRegressor(
                random_state=42
            ),

            "SVR": SVR()
        }

        params = {
            "Linear Regression": {},

            "Ridge": {
                "alpha": [0.01, 0.1, 1.0, 10.0]
            },

            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7]
            },

            "Decision Tree": {
                "max_depth": [3, 5, 8, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },

            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 8, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },

            "Extra Trees": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 8, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },

            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 5],
                "subsample": [0.7, 0.8, 1.0]
            },

            "Hist Gradient Boosting": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_iter": [50, 100, 200],
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": [0.0, 0.1, 1.0]
            },

            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1]
            },

            "SVR": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.05, 0.1]
            }
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )

            params["XGBoost"] = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                objective="regression",
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

            params["LightGBM"] = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 8],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }

        if CATBOOST_AVAILABLE:
            os.makedirs("catboost_info", exist_ok=True)

            models["CatBoost"] = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                random_seed=42,
                verbose=False,
                train_dir="catboost_info"
            )

            params["CatBoost"] = {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5]
            }

        return models, params

    def get_cv_strategy(self, cv_type="timeseries", n_splits=5):
        """
        cv_type options:
        - timeseries: TimeSeriesSplit
        - kfold: random KFold
        """

        if cv_type == "timeseries":
            return TimeSeriesSplit(n_splits=n_splits)

        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

    def train_single_model(
        self,
        model_name,
        model,
        params,
        X_train,
        y_train
    ):
        try:
            model_params = params.get(model_name, {})

            if len(model_params) == 0:
                model.fit(X_train, y_train)
                return model, {}

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=model_params,
                n_iter=5,
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
        cv_type="timeseries"
    ):
        try:
            cv = self.get_cv_strategy(
                cv_type=cv_type,
                n_splits=5
            )

            model_report = {}
            trained_model_store = {}

            for model_name, model in models.items():
                logging.info(f"Starting training for model: {model_name}")
                logging.info(f"CV strategy: {cv_type}")

                fold_models = []
                fold_scalers = []
                fold_mae_scores = []
                fold_r2_scores = []
                fold_best_params = []

                for fold, (train_index, valid_index) in enumerate(cv.split(X)):
                    logging.info(
                        f"{model_name} | Fold {fold + 1} started"
                    )

                    X_train = X.iloc[train_index]
                    X_valid = X.iloc[valid_index]

                    y_train = y[train_index]
                    y_valid = y[valid_index]

                    scaler = StandardScaler()

                    X_train_scaled = scaler.fit_transform(X_train)
                    X_valid_scaled = scaler.transform(X_valid)

                    trained_model, best_params = self.train_single_model(
                        model_name=model_name,
                        model=model,
                        params=params,
                        X_train=X_train_scaled,
                        y_train=y_train
                    )

                    y_pred = trained_model.predict(X_valid_scaled)

                    y_pred = np.clip(y_pred, 0, 16)

                    mae = mean_absolute_error(y_valid, y_pred)
                    r2 = r2_score(y_valid, y_pred)

                    fold_models.append(trained_model)
                    fold_scalers.append(scaler)
                    fold_mae_scores.append(float(mae))
                    fold_r2_scores.append(float(r2))
                    fold_best_params.append(best_params)

                    logging.info(
                        f"{model_name} | Fold {fold + 1} MAE: {mae}"
                    )

                    logging.info(
                        f"{model_name} | Fold {fold + 1} R2: {r2}"
                    )

                avg_mae = float(np.mean(fold_mae_scores))
                avg_r2 = float(np.mean(fold_r2_scores))

                model_report[model_name] = {
                    "avg_mae": avg_mae,
                    "avg_r2": avg_r2,
                    "fold_mae_scores": fold_mae_scores,
                    "fold_r2_scores": fold_r2_scores,
                    "cv_type": cv_type
                }

                trained_model_store[model_name] = {
                    "models": fold_models,
                    "scalers": fold_scalers,
                    "best_params": fold_best_params,
                    "avg_mae": avg_mae,
                    "avg_r2": avg_r2,
                    "cv_type": cv_type
                }

                logging.info(
                    f"{model_name} completed | Avg MAE: {avg_mae} | Avg R2: {avg_r2}"
                )

            return model_report, trained_model_store

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self,
        X,
        y,
        cv_type="timeseries"
    ):
        """
        Advanced model trainer.

        Uses:
        - advanced regression models
        - TimeSeriesSplit or KFold
        - MAE model selection
        """

        try:
            logging.info("Entered advanced model trainer component")
            logging.info(f"Selected CV type: {cv_type}")

            models, params = self.get_models_and_params()

            model_report, trained_model_store = self.evaluate_all_models(
                X=X,
                y=y,
                models=models,
                params=params,
                cv_type=cv_type
            )

            best_model_name = min(
                model_report,
                key=lambda name: model_report[name]["avg_mae"]
            )

            best_model_mae = model_report[best_model_name]["avg_mae"]
            best_model_r2 = model_report[best_model_name]["avg_r2"]

            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Best model MAE: {best_model_mae}")
            logging.info(f"Best model R2: {best_model_r2}")

            best_model_object = trained_model_store[best_model_name]

            final_model_object = {
                "best_model_name": best_model_name,
                "best_model_mae": best_model_mae,
                "best_model_r2": best_model_r2,
                "cv_type": cv_type,
                "models": best_model_object["models"],
                "scalers": best_model_object["scalers"],
                "best_params": best_model_object["best_params"],
                "feature_columns": list(X.columns),
                "model_report": model_report,
                "all_model_results": trained_model_store
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
                "best_model_r2": best_model_r2,
                "cv_type": cv_type,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "number_of_models": len(best_model_object["models"])
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("ModelTrainer is used by train_pipeline.py.")
    print("Run: python -m src.pipeline.train_pipeline")