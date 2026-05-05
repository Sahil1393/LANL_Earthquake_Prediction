# 🌍 LANL Earthquake Prediction (ML & DL Project)

## 📌 Overview

This project aims to predict earthquake occurrences using seismic time-series data from the LANL dataset.

It follows a **modular machine learning & deep learning pipeline architecture**, including data ingestion, transformation, feature engineering, and model training.

---

## 🚀 Project Objectives

* Analyze seismic time-series data
* Build a robust ML & DL pipeline
* Perform feature engineering on signal data
* Train and evaluate multiple models
* Create a scalable and production-ready architecture

---

## 🧠 Problem Statement

Given continuous seismic signal data, predict the **time remaining before the next earthquake (Time-to-Failure - TTF)**.

This is a challenging problem due to:

* Large-scale time-series data
* Noisy and irregular signal patterns
* Complex feature extraction requirements

---

## 🗂️ Project Structure
---

LANL_EARTHQUAKE_PREDICTION/
│
├── artifacts/                 # Generated files (datasets, models, preprocessor)
│   ├── raw.csv
│   ├── transformed_train.csv
│   ├── feature_columns.pkl
│   ├── final_model.pkl
│   ├── model_report.csv
│   ├── deep_learning_model.keras
│   ├── deep_learning_scaler.pkl
│   └── deep_learning_metadata.pkl
│
├── catboost_info/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   ├── learn/
│   └── tmp/
│
├── logs/
│
├── notebook/               # Jupyter notebooks (EDA & experiments)
│   ├── data/
│   │   └── train.csv
│   ├── EDA.ipynb
│   └── Model_train.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   └── deep_learning_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │   └── deep_learning_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   | index.html
|
│__ venv              # Virtual environment
├── app.py
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
---

## 🔄 ML Pipeline Workflow

### 1. Data Ingestion

* Reads seismic dataset
* Splits into train segments
* Stores data for processing

---

### 2. Data Transformation

* Cleans signal data
* Applies normalization/scaling
* Handles large time-series chunks

---

### 3. Feature Engineering

* Extracts statistical features from signals
* Converts raw signals into meaningful inputs
* Saves feature columns (`.pkl`)

---

### 4. Model Training

* Trains ML models (XGBoost, CatBoost)
* Trains Deep Learning models (Keras)
* Evaluates performance
* Saves best model in `artifacts/`

---

## 📊 Input Data

```
notebook/data/train.csv
```

* Contains acoustic signal and time-to-failure values

---

## 📦 Output

```

├── artifacts/
│   ├── raw.csv
│   ├── transformed_train.csv
│   ├── feature_columns.pkl
│   ├── final_model.pkl
│   ├── model_report.csv
│   ├── deep_learning_model.keras
│   ├── deep_learning_scaler.pkl
│   └── deep_learning_metadata.pkl

---

## 🧪 How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/Sahil1393/LANL_Earthquake_Prediction.git
cd LANL_Earthquake_Prediction
```

---

### Step 2: Create virtual environment

```bash
create -p venv python==3.10 -y 
conda activate venv\  # Windows
```

---

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Run Training Pipeline

```bash
python src/pipeline/train_pipeline.py
```

---

### Step 5: Run Prediction

```bash
python src/pipeline/predict_pipeline.py
```

---

## 📌 Key Highlights

* Modular ML + DL pipeline
* Handles large time-series data
* Feature engineering for seismic signals
* Clean and production-ready structure
* Logging and exception handling implemented

---

## ⚠️ Common Issues & Fixes

| Issue                | Solution                 |
| -------------------- | ------------------------ |
| TensorFlow not found | `pip install tensorflow` |
| XGBoost error        | `pip install xgboost`    |
| Import error         | Run from root directory  |
| Memory issue         | Reduce segment size      |

---

## 📚 Future Improvements

* Hyperparameter tuning
* Real-time prediction system
* Model deployment (API)
* Dashboard visualization
* Dockerization

---

## 👥 Authors

**Sahil Gupta** | Enrollment No: 220220357
GitHub: https://github.com/Sahil1393 | LinkedIn: https://www.linkedin.com/in/thatssahil/

**Safa Janjirkar** | Enrollment No: 220221373
GitHub: https://github.com/SafaJanjirkar07 | LinkedIn: https://www.linkedin.com/in/safa-janjirkar-b82164288/

**Shraddha Landge** | Enrollment No: 220220246
GitHub: https://github.com/shraddha365 | LinkedIn: https://www.linkedin.com/in/shraddha-landge

**Prathamesh Maurya** | Enrollment No: 220220178
GitHub: https://github.com/btwitsrich | LinkedIn: https://www.linkedin.com/in/prathameshmaurya/

---


## ⭐ Acknowledgements

* Kaggle LANL Dataset
* Scikit-learn Documentation
* TensorFlow Documentation

---

## 📬 Contact

Feel free to connect for collaboration or queries 🚀
