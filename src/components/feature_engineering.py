import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


class FeatureEngineering:
    def __init__(self):
        pass

    def create_features(self, segment):
        """
        Creates statistical, rolling, chunk-based, and FFT features
        from one acoustic_data segment.
        """

        try:
            if "acoustic_data" not in segment.columns:
                raise ValueError("Input segment must contain acoustic_data column")

            x = segment["acoustic_data"].values.astype(np.float32)

            if len(x) == 0:
                raise ValueError("Input segment is empty")

            features = {}

            # Basic statistical features
            features["mean"] = x.mean()
            features["std"] = x.std()
            features["max"] = x.max()
            features["min"] = x.min()
            features["range"] = x.max() - x.min()
            features["median"] = np.median(x)

            # Absolute value features
            abs_x = np.abs(x)

            features["abs_mean"] = abs_x.mean()
            features["abs_std"] = abs_x.std()
            features["abs_max"] = abs_x.max()

            # Quantile features
            features["q01"] = np.percentile(x, 1)
            features["q05"] = np.percentile(x, 5)
            features["q10"] = np.percentile(x, 10)
            features["q25"] = np.percentile(x, 25)
            features["q50"] = np.percentile(x, 50)
            features["q75"] = np.percentile(x, 75)
            features["q90"] = np.percentile(x, 90)
            features["q95"] = np.percentile(x, 95)
            features["q99"] = np.percentile(x, 99)

            # Energy and signal features
            features["energy"] = np.sum(x ** 2) / len(x)
            features["ptp"] = np.ptp(x)
            features["zero_cross"] = np.mean(np.diff(np.sign(x)) != 0)

            # Trend feature
            features["trend"] = np.polyfit(np.arange(len(x)), x, 1)[0]

            # Distribution features
            series = pd.Series(x)

            features["skew"] = series.skew()
            features["kurtosis"] = series.kurtosis()

            # Rolling window features
            for window in [100, 500, 1000, 5000]:
                rolling_series = series.rolling(window)

                features[f"rolling_mean_{window}"] = rolling_series.mean().mean()
                features[f"rolling_std_{window}"] = rolling_series.std().mean()
                features[f"rolling_max_{window}"] = rolling_series.max().mean()
                features[f"rolling_min_{window}"] = rolling_series.min().mean()

            # Chunk-based features
            chunks = 5
            chunk_size = len(x) // chunks

            chunk_means = []
            chunk_stds = []

            for i in range(chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size

                chunk = x[start:end]

                features[f"chunk_{i}_mean"] = chunk.mean()
                features[f"chunk_{i}_std"] = chunk.std()
                features[f"chunk_{i}_max"] = chunk.max()
                features[f"chunk_{i}_min"] = chunk.min()

                chunk_means.append(chunk.mean())
                chunk_stds.append(chunk.std())

            features["chunk_mean_diff"] = chunk_means[-1] - chunk_means[0]
            features["chunk_std_diff"] = chunk_stds[-1] - chunk_stds[0]

            # FFT frequency-domain features
            fft_values = np.fft.rfft(x)
            fft_magnitude = np.abs(fft_values)

            features["fft_mean"] = fft_magnitude.mean()
            features["fft_std"] = fft_magnitude.std()
            features["fft_max"] = fft_magnitude.max()
            features["fft_min"] = fft_magnitude.min()
            features["fft_median"] = np.median(fft_magnitude)

            # Clean NaN and infinity values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0

            return features

        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_dataframe(
        self,
        df,
        segment_size=150_000,
        step_size=10_000
    ):
        """
        Converts raw acoustic signal data into ML-ready feature rows.

        X = engineered feature dataframe
        y = time_to_failure target array
        """

        try:
            logging.info("Feature dataframe creation started")

            required_columns = ["acoustic_data", "time_to_failure"]

            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            X = []
            y = []

            for start in range(0, len(df) - segment_size, step_size):
                end = start + segment_size

                segment = df.iloc[start:end]

                features = self.create_features(segment)

                X.append(features)
                y.append(segment["time_to_failure"].values[-1])

            X = pd.DataFrame(X)
            y = np.array(y)

            logging.info(f"Feature dataframe created. Shape: {X.shape}")
            logging.info(f"Target array created. Shape: {y.shape}")

            return X, y

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("FeatureEngineering is a helper component.")
    print("Run data_transformation.py or train_pipeline.py instead.")