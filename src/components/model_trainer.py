import os
import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "final_model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X, y):
        try:
            logging.info("Entered model trainer component")

            kf = KFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            params = {
                "objective": "regression",
                "metric": "mae",
                "learning_rate": 0.005,
                "num_leaves": 64,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.5,
                "lambda_l2": 0.5,
                "verbosity": -1
            }

            oof_preds = np.zeros(len(X))

            models = []
            scalers = []
            all_evals = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logging.info(f"========== FOLD {fold + 1} STARTED ==========")

                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]

                y_train = y[train_idx]
                y_val = y[val_idx]

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                evals_result = {}

                train_dataset = lgb.Dataset(
                    X_train_scaled,
                    label=y_train
                )

                valid_dataset = lgb.Dataset(
                    X_val_scaled,
                    label=y_val
                )

                model = lgb.train(
                    params=params,
                    train_set=train_dataset,
                    valid_sets=[
                        train_dataset,
                        valid_dataset
                    ],
                    valid_names=[
                        "train",
                        "valid"
                    ],
                    num_boost_round=4000,
                    callbacks=[
                        lgb.early_stopping(200),
                        lgb.record_evaluation(evals_result)
                    ]
                )

                preds = model.predict(X_val_scaled)
                preds = np.clip(preds, 0, 16)

                oof_preds[val_idx] = preds

                fold_mae = mean_absolute_error(y_val, preds)

                logging.info(f"Fold {fold + 1} MAE: {fold_mae}")

                # Important fixed part
                models.append(model)
                scalers.append(scaler)
                all_evals.append(evals_result)

                logging.info(f"========== FOLD {fold + 1} COMPLETED ==========")

            final_mae = mean_absolute_error(y, oof_preds)

            logging.info(f"Final CV MAE: {final_mae}")

            feature_importance = np.mean(
                [model.feature_importance() for model in models],
                axis=0
            )

            model_object = {
                "models": models,
                "scalers": scalers,
                "feature_columns": list(X.columns),
                "oof_predictions": oof_preds,
                "cv_mae": final_mae,
                "evals_result": all_evals,
                "feature_importance": feature_importance
            }

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_object
            )

            logging.info("Final model object saved successfully")

            return {
                "cv_mae": final_mae,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "number_of_models": len(models)
            }

        except Exception as e:
            raise CustomException(e, sys)