import os
import sys
import gc
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

    transformed_train_file_path: str = os.path.join(
        "artifacts",
        "transformed_train.csv"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.feature_engineering = FeatureEngineering()

    def initiate_data_transformation(
        self,
        raw_data_path,
        segment_size=150_000,
        step_size=10_000,
        chunksize=500_000,
        save_transformed_data=True,
        max_chunks=None
    ):
        """
        Converts partial large LANL data into engineered features.

        This version:
        - Reads artifacts/raw.csv in chunks
        - Creates engineered features
        - Saves feature column names
        - Saves transformed_train.csv for model training
        """

        logging.info("Entered data transformation component")

        try:
            if not os.path.exists(raw_data_path):
                raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

            os.makedirs("artifacts", exist_ok=True)

            logging.info(f"Reading raw data from: {raw_data_path}")
            logging.info(f"Segment size: {segment_size}")
            logging.info(f"Step size: {step_size}")
            logging.info(f"Chunksize: {chunksize}")
            logging.info(f"Save transformed data: {save_transformed_data}")
            logging.info(f"Max chunks: {max_chunks}")

            X_list = []
            y_list = []

            previous_tail = pd.DataFrame()
            feature_columns = None
            chunk_number = 0

            for chunk in pd.read_csv(raw_data_path, chunksize=chunksize):
                chunk_number += 1

                if max_chunks is not None and chunk_number > max_chunks:
                    logging.info(f"Stopping transformation after {max_chunks} chunks.")
                    break

                logging.info(
                    f"Processing transformation chunk {chunk_number}, "
                    f"shape: {chunk.shape}"
                )

                if not previous_tail.empty:
                    chunk = pd.concat(
                        [previous_tail, chunk],
                        axis=0,
                        ignore_index=True
                    )

                if len(chunk) < segment_size:
                    previous_tail = chunk.copy()
                    continue

                X_chunk, y_chunk = self.feature_engineering.create_feature_dataframe(
                    df=chunk,
                    segment_size=segment_size,
                    step_size=step_size
                )

                if X_chunk.empty:
                    logging.warning(
                        f"Chunk {chunk_number} produced empty features. Skipping."
                    )

                    previous_tail = chunk.tail(segment_size).copy()

                    del chunk
                    gc.collect()
                    continue

                if feature_columns is None:
                    feature_columns = list(X_chunk.columns)

                X_chunk = X_chunk[feature_columns]

                X_list.append(X_chunk)
                y_list.append(pd.Series(y_chunk))

                logging.info(
                    f"Chunk {chunk_number} transformed successfully. "
                    f"X_chunk shape: {X_chunk.shape}, "
                    f"y_chunk shape: {len(y_chunk)}"
                )

                previous_tail = chunk.tail(segment_size).copy()

                del chunk
                del X_chunk
                del y_chunk

                gc.collect()

            if len(X_list) == 0:
                raise ValueError(
                    "Feature dataframe is empty. "
                    "Try increasing ingestion max_chunks, increasing chunksize, "
                    "or reducing segment_size."
                )

            X = pd.concat(X_list, axis=0, ignore_index=True)
            y = pd.concat(y_list, axis=0, ignore_index=True)

            if X.empty:
                raise ValueError(
                    "Final feature dataframe is empty. "
                    "Increase data size or reduce segment_size."
                )

            if len(X) != len(y):
                raise ValueError(
                    f"X and y row mismatch. X rows: {len(X)}, y rows: {len(y)}"
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

            if save_transformed_data:
                transformed_df = X.copy()
                transformed_df["target"] = y.values

                transformed_df.to_csv(
                    self.data_transformation_config.transformed_train_file_path,
                    index=False
                )

                logging.info(
                    f"Transformed training data saved successfully at: "
                    f"{self.data_transformation_config.transformed_train_file_path}"
                )

            logging.info(f"Final X shape: {X.shape}")
            logging.info(f"Final y shape: {y.shape}")

            return (
                X,
                y,
                self.data_transformation_config.feature_columns_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()

    raw_data_path = "artifacts/raw.csv"

    print("Using raw data path:", raw_data_path)
    print("File exists:", os.path.exists(raw_data_path))

    if os.path.exists(raw_data_path):
        print(
            "File size MB:",
            round(os.path.getsize(raw_data_path) / (1024 * 1024), 2)
        )

    X, y, feature_columns_path = obj.initiate_data_transformation(
        raw_data_path=raw_data_path,
        segment_size=150_000,
        step_size=10_000,
        chunksize=500_000,
        save_transformed_data=True,
        max_chunks=None
    )

    print("Data transformation completed successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature columns saved at:", feature_columns_path)