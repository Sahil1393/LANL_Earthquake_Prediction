import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """
        Full training pipeline:

        1. Data ingestion:
           Reads partial data from notebook/data/train.csv
           Saves artifacts/raw.csv

        2. Data transformation:
           Reads artifacts/raw.csv
           Creates features and target
           Saves artifacts/transformed_train.csv
           Saves artifacts/feature_columns.pkl

        3. Model training:
           Trains multiple models
           Saves artifacts/final_model.pkl
           Saves artifacts/model_report.csv
        """

        try:
            logging.info("Training pipeline started")

            # Step 1: Data ingestion
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion()

            raw_data_path = data_ingestion.initiate_data_ingestion(
                input_file_path="notebook/data/train.csv",
                chunksize=500_000,
                max_chunks=30
            )

            logging.info(f"Data ingestion completed. Raw data path: {raw_data_path}")

            # Step 2: Data transformation
            logging.info("Starting data transformation")

            data_transformation = DataTransformation()

            X, y, feature_columns_path = data_transformation.initiate_data_transformation(
                raw_data_path=raw_data_path,
                segment_size=150_000,
                step_size=10_000,
                chunksize=500_000,
                save_transformed_data=True,
                max_chunks=None
            )

            logging.info(f"Data transformation completed. X shape: {X.shape}")
            logging.info(f"y shape: {y.shape}")
            logging.info(f"Feature columns path: {feature_columns_path}")

            # Step 3: Model training
            logging.info("Starting model training")

            model_trainer = ModelTrainer()

            training_result = model_trainer.initiate_model_trainer(
                X=X,
                y=y,
                fast_mode=True,
                n_splits=5,
                n_iter=8
            )

            logging.info("Model training completed")
            logging.info(f"Training result: {training_result}")

            print("Training pipeline completed successfully")
            print("Best model:", training_result["best_model_name"])
            print("Best MAE:", training_result["best_model_mae"])
            print("Best R2:", training_result["best_model_r2"])
            print("Training samples:", training_result["training_samples"])
            print("Number of features:", training_result["number_of_features"])
            print("Model saved at:", training_result["model_path"])
            print("Model report saved at:", training_result["model_report_path"])

            return training_result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()