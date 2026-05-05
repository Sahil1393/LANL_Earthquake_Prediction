import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.feature_engineering import FeatureEngineering


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "final_model.pkl")
        self.feature_columns_path = os.path.join("artifacts", "feature_columns.pkl")

        self.feature_engineering = FeatureEngineering()

    def load_model_artifacts(self):
        """
        Loads trained model object and saved feature columns.
        """

        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}. "
                    "Run train_pipeline.py first."
                )

            if not os.path.exists(self.feature_columns_path):
                raise FileNotFoundError(
                    f"Feature columns file not found: {self.feature_columns_path}. "
                    "Run data_transformation.py first."
                )

            model_object = load_object(self.model_path)
            feature_columns = load_object(self.feature_columns_path)

            logging.info("Model artifacts loaded successfully")

            return model_object, feature_columns

        except Exception as e:
            raise CustomException(e, sys)

    def prepare_input_features(
        self,
        input_data_path,
        segment_size=150_000,
        step_size=150_000
    ):
        """
        Converts raw test/input acoustic data into feature dataframe.

        For prediction:
        - If input has acoustic_data only, prediction still works.
        - If input does not have time_to_failure, dummy target is created
          because FeatureEngineering expects that column.
        """

        try:
            if not os.path.exists(input_data_path):
                raise FileNotFoundError(f"Input data file not found: {input_data_path}")

            logging.info(f"Reading prediction input data from: {input_data_path}")

            df = pd.read_csv(input_data_path)

            if "acoustic_data" not in df.columns:
                raise ValueError("Input file must contain acoustic_data column")

            if "time_to_failure" not in df.columns:
                df["time_to_failure"] = 0

            X, _ = self.feature_engineering.create_feature_dataframe(
                df=df,
                segment_size=segment_size,
                step_size=step_size
            )

            if X.empty:
                raise ValueError(
                    "No prediction features created. "
                    "Input file must contain at least segment_size rows."
                )

            logging.info(f"Prediction feature dataframe created. Shape: {X.shape}")

            return X

        except Exception as e:
            raise CustomException(e, sys)

    def align_features(self, X, feature_columns):
        """
        Aligns prediction features with training feature columns.
        Missing columns are filled with 0.
        Extra columns are removed.
        """

        try:
            X = X.copy()

            for column in feature_columns:
                if column not in X.columns:
                    X[column] = 0

            X = X[feature_columns]

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            logging.info("Prediction features aligned successfully")

            return X

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data_path):
        """
        Predicts time_to_failure using trained model.

        The saved model contains multiple fold models.
        Final prediction = average prediction from all fold models.
        """

        try:
            logging.info("Prediction pipeline started")

            model_object, feature_columns = self.load_model_artifacts()

            X = self.prepare_input_features(
                input_data_path=input_data_path,
                segment_size=150_000,
                step_size=150_000
            )

            X = self.align_features(X, feature_columns)

            models = model_object["models"]
            scalers = model_object.get("scalers", [])

            predictions = []

            for index, model in enumerate(models):
                scaler = scalers[index] if index < len(scalers) else None

                if scaler is not None:
                    X_model = scaler.transform(X)
                else:
                    X_model = X

                pred = model.predict(X_model)
                predictions.append(pred)

            final_prediction = np.mean(predictions, axis=0)

            final_prediction = np.clip(final_prediction, 0, None)

            logging.info("Prediction completed successfully")

            return final_prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Helper class for single segment prediction from acoustic_data array/list.
    """

    def __init__(self, acoustic_data):
        self.acoustic_data = acoustic_data

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame(
                {
                    "acoustic_data": self.acoustic_data,
                    "time_to_failure": 0
                }
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = PredictPipeline()

    input_data_path = "artifacts/raw.csv"

    predictions = pipeline.predict(input_data_path)

    print("Prediction completed successfully")
    print("Predictions:")
    print(predictions)