import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.deep_learning_trainer import DeepLearningTrainer


class DeepLearningPipeline:
    def __init__(
        self,
        input_file_path,
        nrows=5_000_000,
        model_type="cnn_lstm",
        sequence_length=150_000,
        step_size=25_000,
        epochs=20,
        batch_size=8
    ):
        self.input_file_path = input_file_path
        self.nrows = nrows
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size

    def run_pipeline(self):
        try:
            logging.info("Deep learning pipeline started")

            print("=" * 70)
            print("DEEP LEARNING PIPELINE STARTED")
            print("=" * 70)

            print("[1/2] Data ingestion started...")

            data_ingestion = DataIngestion()

            raw_data_path = data_ingestion.initiate_data_ingestion(
                input_file_path=self.input_file_path,
                nrows=self.nrows
            )

            print("Data ingestion completed.")
            print("Raw data saved at:", raw_data_path)

            print("-" * 70)
            print("[2/2] Deep learning training started...")

            trainer = DeepLearningTrainer()

            result = trainer.initiate_deep_learning_training(
                raw_data_path=raw_data_path,
                model_type=self.model_type,
                sequence_length=self.sequence_length,
                step_size=self.step_size,
                epochs=self.epochs,
                batch_size=self.batch_size
            )

            print("=" * 70)
            print("DEEP LEARNING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(result)

            logging.info("Deep learning pipeline completed successfully")

            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = DeepLearningPipeline(
        input_file_path="notebook/data/train.csv",
        nrows=5_000_000,
        model_type="cnn_lstm",
        sequence_length=150_000,
        step_size=25_000,
        epochs=20,
        batch_size=8
    )

    pipeline.run_pipeline()