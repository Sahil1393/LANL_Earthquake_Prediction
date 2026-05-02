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
        Converts raw LANL earthquake data into ML-ready features.

        Input:
            raw_data_path: path of raw.csv from data ingestion

        Output:
            X: feature DataFrame
            y: target array
            feature_columns_file_path: saved feature column path
        """

        logging.info("Entered the data transformation component")

        try:
            logging.info(f"Reading raw data from: {raw_data_path}")

            df = pd.read_csv(raw_data_path)

            logging.info(f"Raw data loaded successfully. Shape: {df.shape}")

            X, y = self.feature_engineering.create_feature_dataframe(
                df=df,
                segment_size=segment_size,
                step_size=step_size
            )

            logging.info(f"Feature DataFrame created. Shape: {X.shape}")
            logging.info(f"Target array created. Shape: {y.shape}")

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