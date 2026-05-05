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
            features["mean"] = np.mean(x)
            features["std"] = np.std(x)
            features["max"] = np.max(x)
            features["min"] = np.min(x)
            features["range"] = np.max(x) - np.min(x)
            features["median"] = np.median(x)

            # Absolute value features
            abs_x = np.abs(x)

            features["abs_mean"] = np.mean(abs_x)
            features["abs_std"] = np.std(abs_x)
            features["abs_max"] = np.max(abs_x)
            features["abs_min"] = np.min(abs_x)

            # Quantile features
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                features[f"q{q:02d}"] = np.percentile(x, q)

            # Absolute quantile features
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                features[f"abs_q{q:02d}"] = np.percentile(abs_x, q)

            # Energy and signal features
            features["energy"] = np.sum(x ** 2) / len(x)
            features["ptp"] = np.ptp(x)
            features["zero_cross"] = np.mean(np.diff(np.sign(x)) != 0)

            # Trend feature
            idx = np.arange(len(x), dtype=np.float32)
            features["trend"] = np.polyfit(idx, x, 1)[0]

            # Distribution features
            series = pd.Series(x)

            features["skew"] = series.skew()
            features["kurtosis"] = series.kurtosis()

            # Difference features
            diff_x = np.diff(x)

            if len(diff_x) > 0:
                features["diff_mean"] = np.mean(diff_x)
                features["diff_std"] = np.std(diff_x)
                features["diff_max"] = np.max(diff_x)
                features["diff_min"] = np.min(diff_x)
                features["diff_abs_mean"] = np.mean(np.abs(diff_x))
            else:
                features["diff_mean"] = 0
                features["diff_std"] = 0
                features["diff_max"] = 0
                features["diff_min"] = 0
                features["diff_abs_mean"] = 0

            # Rolling window features
            for window in [100, 500, 1000, 5000, 10000]:
                if len(series) >= window:
                    rolling_series = series.rolling(window=window)

                    features[f"rolling_mean_{window}"] = (
                        rolling_series.mean().dropna().mean()
                    )
                    features[f"rolling_std_{window}"] = (
                        rolling_series.std().dropna().mean()
                    )
                    features[f"rolling_max_{window}"] = (
                        rolling_series.max().dropna().mean()
                    )
                    features[f"rolling_min_{window}"] = (
                        rolling_series.min().dropna().mean()
                    )
                else:
                    features[f"rolling_mean_{window}"] = 0
                    features[f"rolling_std_{window}"] = 0
                    features[f"rolling_max_{window}"] = 0
                    features[f"rolling_min_{window}"] = 0

            # Chunk-based features
            chunks = 5
            chunk_size = len(x) // chunks

            chunk_means = []
            chunk_stds = []

            for i in range(chunks):
                start = i * chunk_size

                if i == chunks - 1:
                    end = len(x)
                else:
                    end = (i + 1) * chunk_size

                chunk_data = x[start:end]

                if len(chunk_data) == 0:
                    chunk_mean = 0
                    chunk_std = 0
                    chunk_max = 0
                    chunk_min = 0
                else:
                    chunk_mean = np.mean(chunk_data)
                    chunk_std = np.std(chunk_data)
                    chunk_max = np.max(chunk_data)
                    chunk_min = np.min(chunk_data)

                features[f"chunk_{i}_mean"] = chunk_mean
                features[f"chunk_{i}_std"] = chunk_std
                features[f"chunk_{i}_max"] = chunk_max
                features[f"chunk_{i}_min"] = chunk_min

                chunk_means.append(chunk_mean)
                chunk_stds.append(chunk_std)

            features["chunk_mean_diff"] = chunk_means[-1] - chunk_means[0]
            features["chunk_std_diff"] = chunk_stds[-1] - chunk_stds[0]

            # FFT frequency-domain features
            fft_values = np.fft.rfft(x)
            fft_magnitude = np.abs(fft_values)

            features["fft_mean"] = np.mean(fft_magnitude)
            features["fft_std"] = np.std(fft_magnitude)
            features["fft_max"] = np.max(fft_magnitude)
            features["fft_min"] = np.min(fft_magnitude)
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

            if len(df) < segment_size:
                logging.warning(
                    f"Data length {len(df)} is smaller than segment_size "
                    f"{segment_size}. Returning empty X and y."
                )
                return pd.DataFrame(), np.array([])

            X = []
            y = []

            total_segments = 0

            # +1 is important so the final possible segment is included
            for start in range(0, len(df) - segment_size + 1, step_size):
                end = start + segment_size

                segment = df.iloc[start:end]

                features = self.create_features(segment)

                X.append(features)
                y.append(segment["time_to_failure"].values[-1])

                total_segments += 1

            X = pd.DataFrame(X)
            y = np.array(y)

            logging.info(f"Total segments created: {total_segments}")
            logging.info(f"Feature dataframe created. Shape: {X.shape}")
            logging.info(f"Target array created. Shape: {y.shape}")

            return X, y

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("FeatureEngineering is a helper component.")
    print("Run data_ingestion.py, data_transformation.py, or train_pipeline.py instead.")