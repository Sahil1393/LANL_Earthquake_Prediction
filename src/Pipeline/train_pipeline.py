import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(
        self,
        input_file_path,
        nrows=500_000,
        segment_size=150_000,
        step_size=10_000,
        cv_type="timeseries"
    ):
        self.input_file_path = input_file_path
        self.nrows = nrows
        self.segment_size = segment_size
        self.step_size = step_size
        self.cv_type = cv_type

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            print("=" * 70)
            print("TRAINING PIPELINE STARTED")
            print("=" * 70)

            print("[1/3] Data ingestion started...")

            data_ingestion = DataIngestion()

            raw_data_path = data_ingestion.initiate_data_ingestion(
                input_file_path=self.input_file_path,
                nrows=self.nrows
            )

            print("Data ingestion completed.")
            print(f"Raw data saved at: {raw_data_path}")

            print("-" * 70)
            print("[2/3] Data transformation and feature engineering started...")

            data_transformation = DataTransformation()

            X, y, feature_columns_path = data_transformation.initiate_data_transformation(
                raw_data_path=raw_data_path,
                segment_size=self.segment_size,
                step_size=self.step_size
            )

            print("Data transformation completed.")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"Feature columns saved at: {feature_columns_path}")

            print("-" * 70)
            print("[3/3] Model training started...")

            model_trainer = ModelTrainer()

            try:
                result = model_trainer.initiate_model_trainer(
                    X=X,
                    y=y,
                    cv_type=self.cv_type
                )
            except TypeError:
                result = model_trainer.initiate_model_trainer(
                    X=X,
                    y=y
                )

            print("Model training completed.")

            print("=" * 70)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(result)

            logging.info("Training pipeline completed successfully")

            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline(
        input_file_path="notebook/data/train.csv",
        nrows=500_000,
        segment_size=150_000,
        step_size=10_000,
        cv_type="timeseries"
    )

    pipeline.run_pipeline()