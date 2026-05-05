import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        LSTM,
        GRU,
        Dense,
        Dropout,
        BatchNormalization,
        Flatten,
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True

except Exception:
    TENSORFLOW_AVAILABLE = False


@dataclass
class DeepLearningTrainerConfig:
    model_file_path: str = os.path.join(
        "artifacts",
        "deep_learning_model.keras"
    )

    scaler_file_path: str = os.path.join(
        "artifacts",
        "deep_learning_scaler.pkl"
    )

    metadata_file_path: str = os.path.join(
        "artifacts",
        "deep_learning_metadata.pkl"
    )


class DeepLearningTrainer:
    def __init__(self):
        self.config = DeepLearningTrainerConfig()

    def check_tensorflow(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. "
                "Install it using: pip install tensorflow"
            )

    def create_sequences(
        self,
        df,
        sequence_length=150_000,
        step_size=25_000
    ):
        """
        Converts raw acoustic_data into deep learning sequences.

        Input dataframe columns:
        - acoustic_data
        - time_to_failure

        Output:
        X shape = samples, sequence_length, 1
        y shape = samples
        """

        try:
            logging.info("Creating deep learning sequences")

            required_columns = ["acoustic_data", "time_to_failure"]

            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            X = []
            y = []

            total_rows = len(df)

            for start in range(0, total_rows - sequence_length, step_size):
                end = start + sequence_length

                segment = df.iloc[start:end]

                signal = segment["acoustic_data"].values.astype(np.float32)
                target = segment["time_to_failure"].values[-1]

                X.append(signal)
                y.append(target)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            if len(X) == 0:
                raise ValueError(
                    "No sequences created. "
                    "Increase nrows or reduce sequence_length."
                )

            X = X.reshape(X.shape[0], X.shape[1], 1)

            logging.info(f"Deep learning X shape: {X.shape}")
            logging.info(f"Deep learning y shape: {y.shape}")

            print("Deep learning sequences created")
            print("X shape:", X.shape)
            print("y shape:", y.shape)

            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def scale_sequences(self, X_train, X_test):
        """
        Scales 3D sequence data using StandardScaler.

        Original shape:
        samples, sequence_length, 1

        Temporarily converted to:
        samples * sequence_length, 1
        """

        try:
            scaler = StandardScaler()

            n_train, seq_len, n_features = X_train.shape
            n_test = X_test.shape[0]

            X_train_2d = X_train.reshape(-1, n_features)
            X_test_2d = X_test.reshape(-1, n_features)

            X_train_scaled = scaler.fit_transform(X_train_2d)
            X_test_scaled = scaler.transform(X_test_2d)

            X_train_scaled = X_train_scaled.reshape(
                n_train,
                seq_len,
                n_features
            )

            X_test_scaled = X_test_scaled.reshape(
                n_test,
                seq_len,
                n_features
            )

            return X_train_scaled, X_test_scaled, scaler

        except Exception as e:
            raise CustomException(e, sys)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),

            LSTM(64, return_sequences=True),
            Dropout(0.3),

            LSTM(32),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mae"]
        )

        return model

    def build_gru_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),

            GRU(64, return_sequences=True),
            Dropout(0.3),

            GRU(32),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mae"]
        )

        return model

    def build_cnn_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),

            Conv1D(32, kernel_size=7, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Conv1D(64, kernel_size=7, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Conv1D(128, kernel_size=5, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Conv1D(256, kernel_size=5, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Flatten(),

            Dense(128, activation="relu"),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mae"]
        )

        return model

    def build_cnn_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),

            Conv1D(32, kernel_size=7, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Conv1D(64, kernel_size=7, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            Conv1D(128, kernel_size=5, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),

            LSTM(64, return_sequences=False),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mae"]
        )

        return model

    def get_model(self, model_type, input_shape):
        if model_type == "lstm":
            return self.build_lstm_model(input_shape)

        if model_type == "gru":
            return self.build_gru_model(input_shape)

        if model_type == "cnn":
            return self.build_cnn_model(input_shape)

        if model_type == "cnn_lstm":
            return self.build_cnn_lstm_model(input_shape)

        raise ValueError(
            "Invalid model_type. Use: lstm, gru, cnn, or cnn_lstm"
        )

    def initiate_deep_learning_training(
        self,
        raw_data_path,
        model_type="cnn_lstm",
        sequence_length=150_000,
        step_size=25_000,
        epochs=20,
        batch_size=8
    ):
        """
        Main deep learning training function.

        model_type options:
        - lstm
        - gru
        - cnn
        - cnn_lstm
        """

        try:
            self.check_tensorflow()

            logging.info("Deep learning training started")

            print("=" * 70)
            print("DEEP LEARNING TRAINING STARTED")
            print("=" * 70)

            print("Reading raw data from:", raw_data_path)

            df = pd.read_csv(raw_data_path)

            print("Raw data shape:", df.shape)

            X, y = self.create_sequences(
                df=df,
                sequence_length=sequence_length,
                step_size=step_size
            )

            if len(X) < 10:
                raise ValueError(
                    "Not enough sequences for deep learning. "
                    "Increase nrows or reduce step_size."
                )

            print("Splitting data into train and test sets...")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                shuffle=False
            )

            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_train shape:", y_train.shape)
            print("y_test shape:", y_test.shape)

            print("Scaling sequences...")

            X_train_scaled, X_test_scaled, scaler = self.scale_sequences(
                X_train,
                X_test
            )

            input_shape = (
                X_train_scaled.shape[1],
                X_train_scaled.shape[2]
            )

            print("Input shape:", input_shape)
            print("Building model:", model_type)

            model = self.get_model(
                model_type=model_type,
                input_shape=input_shape
            )

            model.summary()

            os.makedirs("artifacts", exist_ok=True)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                ),

                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                ),

                ModelCheckpoint(
                    filepath=self.config.model_file_path,
                    monitor="val_loss",
                    save_best_only=True
                )
            ]

            print("Training deep learning model...")

            history = model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            print("Generating predictions...")

            predictions = model.predict(X_test_scaled).flatten()
            predictions = np.clip(predictions, 0, 16)

            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            save_object(
                file_path=self.config.scaler_file_path,
                obj=scaler
            )

            metadata = {
                "model_type": model_type,
                "sequence_length": sequence_length,
                "step_size": step_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "mae": mae,
                "r2_score": r2,
                "history": history.history
            }

            save_object(
                file_path=self.config.metadata_file_path,
                obj=metadata
            )

            logging.info(f"Deep learning model MAE: {mae}")
            logging.info(f"Deep learning model R2: {r2}")

            print("=" * 70)
            print("DEEP LEARNING TRAINING COMPLETED")
            print("=" * 70)
            print("Model type:", model_type)
            print("MAE:", mae)
            print("R2 score:", r2)
            print("Model saved at:", self.config.model_file_path)
            print("Scaler saved at:", self.config.scaler_file_path)
            print("Metadata saved at:", self.config.metadata_file_path)

            return {
                "model_type": model_type,
                "mae": mae,
                "r2_score": r2,
                "model_path": self.config.model_file_path,
                "scaler_path": self.config.scaler_file_path,
                "metadata_path": self.config.metadata_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = DeepLearningTrainer()

    result = trainer.initiate_deep_learning_training(
        raw_data_path="artifacts/raw.csv",
        model_type="cnn_lstm",
        sequence_length=150_000,
        step_size=25_000,
        epochs=20,
        batch_size=8
    )

    print(result)