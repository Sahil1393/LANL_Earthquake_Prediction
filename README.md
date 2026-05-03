# LANL Earthquake Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-ff69b4)

A machine learning project designed to predict the **time remaining before the next laboratory earthquake** (in seconds) from real-time seismic acoustic signal data. The project uses a LightGBM ensemble trained with 5-fold cross-validation, achieving a CV Mean Absolute Error (MAE) of approximately 0.106. 

This repository contains the full end-to-end ML pipeline, from data ingestion and feature engineering to a production-ready FastAPI application for serving predictions.

## Features

- **Robust Feature Engineering:** Extracts meaningful statistical and rolling features from raw seismic acoustic data.
- **Model Training Pipeline:** Includes data ingestion, transformation, feature engineering, and a model trainer that uses 5-fold KFold cross-validation with a LightGBM regressor.
- **FastAPI Serving:** A highly performant REST API to serve single and batch predictions.
- **Web UI:** A simple web interface for interacting with the prediction model.
- **Modular Codebase:** Clean, maintainable structure utilizing Object-Oriented Programming principles.

## Tech Stack

- **Data Science:** Python, Pandas, NumPy, Scikit-Learn
- **Machine Learning:** LightGBM, XGBoost, CatBoost
- **API Framework:** FastAPI, Uvicorn
- **Templating:** Jinja2

## Project Structure

```text
LANL_Earthquake_Prediction/
├── app.py                     # FastAPI application and routes
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── artifacts/                 # Saved models and preprocessors (generated)
├── notebook/                  # Jupyter notebooks for EDA and experimentation
├── templates/                 # HTML templates for the Web UI
└── src/                       # Main source code
    ├── components/            # ML Pipeline components
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── feature_engineering.py
    │   └── model_trainer.py
    ├── Pipeline/              # Pipeline orchestrators
    │   ├── train_pipeline.py
    │   └── predict_pipeline.py
    ├── exception.py           # Custom exception handling
    ├── logger.py              # Application logging setup
    └── utils.py               # Helper functions
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LANL_Earthquake_Prediction.git
   cd LANL_Earthquake_Prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

Before starting the API, you may need to train the model to generate the necessary artifacts (if not already present).

```bash
python -m src.Pipeline.train_pipeline
```

### 2. Running the API

Start the FastAPI application using Uvicorn:

```bash
python app.py
# or
uvicorn app:app --host 127.0.0.1 --port 8080 --reload
```

The API will be available at `http://127.0.0.1:8080`.

## API Endpoints

- `GET /` : Web UI for predictions.
- `GET /health` : Check API health, model load status, and uptime.
- `GET /model/info` : Get metadata about the trained model, such as CV MAE, number of folds, and features used.
- `POST /predict` : Predict the time to failure from a single segment of acoustic data (minimum 100 values required).
- `POST /predict/batch` : Predict the time to failure for multiple acoustic data segments in a single request.

### Example: Predicting Time to Failure

**POST** `/predict`
```json
{
  "acoustic_data": [12.0, 14.5, 13.2, 15.1, ...] // At least 100 values
}
```

**Response**
```json
{
  "time_to_failure_seconds": 3.456,
  "input_length": 150000,
  "model_folds": 5
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.