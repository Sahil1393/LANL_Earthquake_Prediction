import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, input_file_path, nrows=500_000):
        """
        Reads LANL earthquake train.csv.

        Required columns:
        - acoustic_data
        - time_to_failure
        """

        logging.info("Entered data ingestion component")

        try:
            logging.info(f"Reading data from: {input_file_path}")

            df = pd.read_csv(input_file_path, nrows=nrows)

            logging.info(f"Dataset loaded successfully. Shape: {df.shape}")

            required_columns = ["acoustic_data", "time_to_failure"]

            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info(
                f"Raw data saved successfully at: "
                f"{self.ingestion_config.raw_data_path}"
            )

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    raw_data_path = obj.initiate_data_ingestion(
        input_file_path="train.csv",
        nrows=500_000
    )

    print("Data ingestion completed successfully")
    print("Raw data saved at:", raw_data_path)