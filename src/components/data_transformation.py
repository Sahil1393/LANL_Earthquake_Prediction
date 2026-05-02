import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.feature_engineering import FeatureEngineering


@dataclass
class DataTransformationConfig:
    feature_columns_file_path: str = os.path.join(
        "artifacts",
        "feature_columns.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.feature_engineering = FeatureEngineering()

    def initiate_data_transformation(
        self,
        raw_data_path,
        segment_size=150_000,
        step_size=10_000
    ):
        """
        Converts raw data into model-ready X and y.
        """

        logging.info("Entered data transformation component")

        try:
            logging.info(f"Reading raw data from: {raw_data_path}")

            df = pd.read_csv(raw_data_path)

            logging.info(f"Raw data loaded successfully. Shape: {df.shape}")

            X, y = self.feature_engineering.create_feature_dataframe(
                df=df,
                segment_size=segment_size,
                step_size=step_size
            )

            if X.empty:
                raise ValueError(
                    "Feature dataframe is empty. "
                    "Increase nrows or reduce segment_size."
                )

            feature_columns = list(X.columns)

            save_object(
                file_path=self.data_transformation_config.feature_columns_file_path,
                obj=feature_columns
            )

            logging.info(
                f"Feature columns saved successfully at: "
                f"{self.data_transformation_config.feature_columns_file_path}"
            )

            return (
                X,
                y,
                self.data_transformation_config.feature_columns_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()

    X, y, feature_columns_path = obj.initiate_data_transformation(
        raw_data_path="artifacts/raw.csv",
        segment_size=150_000,
        step_size=10_000
    )

    print("Data transformation completed successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature columns saved at:", feature_columns_path)