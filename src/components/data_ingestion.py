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

    def initiate_data_ingestion(
        self,
        input_file_path,
        chunksize=500_000,
        max_chunks=5
    ):
        """
        Reads partial large LANL earthquake data using chunks.

        This does not use the full 9GB dataset.
        It reads only:

            chunksize * max_chunks rows

        Example:
            chunksize=500_000, max_chunks=5
            means 2,500,000 rows.

        Required columns:
        - acoustic_data
        - time_to_failure
        """

        logging.info("Entered data ingestion component")

        try:
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"Input file not found: {input_file_path}")

            logging.info(f"Reading data from: {input_file_path}")
            logging.info(f"Chunksize: {chunksize}")
            logging.info(f"Max chunks: {max_chunks}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            required_columns = ["acoustic_data", "time_to_failure"]

            first_write = True
            total_rows = 0
            chunk_number = 0

            for chunk in pd.read_csv(input_file_path, chunksize=chunksize):
                chunk_number += 1

                if max_chunks is not None and chunk_number > max_chunks:
                    logging.info(f"Stopping ingestion after {max_chunks} chunks.")
                    break

                logging.info(
                    f"Processing ingestion chunk {chunk_number}, shape: {chunk.shape}"
                )

                for column in required_columns:
                    if column not in chunk.columns:
                        raise ValueError(f"Missing required column: {column}")

                chunk.to_csv(
                    self.ingestion_config.raw_data_path,
                    mode="w" if first_write else "a",
                    index=False,
                    header=first_write
                )

                first_write = False
                total_rows += len(chunk)

                logging.info(
                    f"Chunk {chunk_number} saved. Total rows saved: {total_rows}"
                )

            if total_rows == 0:
                raise ValueError(
                    "No rows were saved. Increase max_chunks or check input file."
                )

            logging.info(
                f"Raw partial data saved successfully at: "
                f"{self.ingestion_config.raw_data_path}"
            )

            logging.info(f"Total rows saved: {total_rows}")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    raw_data_path = obj.initiate_data_ingestion(
        input_file_path="notebook/data/train.csv",
        chunksize=500_000,
        max_chunks=30
    )

    print("Data ingestion completed successfully")
    print("Raw data saved at:", raw_data_path)