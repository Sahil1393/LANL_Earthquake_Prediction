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
        nrows=5_000_000,
        segment_size=150_000,
        step_size=10_000
    ):
        self.input_file_path = input_file_path
        self.nrows = nrows
        self.segment_size = segment_size
        self.step_size = step_size

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            data_ingestion = DataIngestion()

            raw_data_path = data_ingestion.initiate_data_ingestion(
                input_file_path=self.input_file_path,
                nrows=self.nrows
            )

            data_transformation = DataTransformation()

            X, y, feature_columns_path = data_transformation.initiate_data_transformation(
                raw_data_path=raw_data_path,
                segment_size=self.segment_size,
                step_size=self.step_size
            )

            model_trainer = ModelTrainer()

            result = model_trainer.initiate_model_trainer(
                X=X,
                y=y
            )

            logging.info("Training pipeline completed successfully")

            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline(
        input_file_path="train.csv",
        nrows=5_000_000,
        segment_size=150_000,
        step_size=10_000
    )

    result = pipeline.run_pipeline()

    print(result)