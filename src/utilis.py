import os
import sys
import pickle

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object as a .pkl file.

    Example:
    - model
    - scaler
    - feature column list
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a saved .pkl file.

    Example:
    - final_model.pkl
    - feature_columns.pkl
    """

    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)