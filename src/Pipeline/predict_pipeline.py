import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/final_model.pkl"

    def predict(self, acoustic_data):
        """
        Predicts time_to_failure.

        acoustic_data can be:
        - list
        - numpy array
        - pandas Series
        - pandas DataFrame with acoustic_data column
        """

        try:
            logging.info("Prediction pipeline started")

            model_object = load_object(self.model_path)

            models = model_object["models"]
            scalers = model_object["scalers"]
            feature_columns = model_object["feature_columns"]

            if len(models) == 0:
                raise ValueError("No trained models found inside final_model.pkl")

            if isinstance(acoustic_data, pd.DataFrame):
                segment = acoustic_data.copy()

                if "acoustic_data" not in segment.columns:
                    raise ValueError(
                        "Input DataFrame must contain acoustic_data column"
                    )

            else:
                segment = pd.DataFrame({
                    "acoustic_data": np.array(acoustic_data)
                })

            data_transformation = DataTransformation()

            features = data_transformation.create_features(segment)

            feature_df = pd.DataFrame([features])

            feature_df = feature_df[feature_columns]

            fold_predictions = []

            for model, scaler in zip(models, scalers):
                feature_scaled = scaler.transform(feature_df)

                pred = model.predict(feature_scaled)
                fold_predictions.append(pred[0])

            final_prediction = np.mean(fold_predictions)
            final_prediction = np.clip(final_prediction, 0, 16)

            logging.info(f"Prediction completed: {final_prediction}")

            return final_prediction

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_df = pd.read_csv("train.csv", nrows=150_000)

    predict_pipeline = PredictPipeline()

    prediction = predict_pipeline.predict(sample_df)

    print("Predicted time_to_failure:", prediction)