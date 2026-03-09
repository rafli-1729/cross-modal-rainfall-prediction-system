# Cross-Modal Rainfall Prediction System

An end-to-end machine learning system for forecasting **daily rainfall across multiple locations in Singapore** by converting rainfall charts into structured data and training a predictive model.

This project demonstrates a **cross-modal machine learning pipeline**, where information is extracted from **chart images** and transformed into **tabular time-series data** used for rainfall forecasting.

The system includes:

* Computer vision pipeline for rainfall chart extraction
* Dataset construction and feature engineering
* Machine learning forecasting model
* FastAPI inference service
* Streamlit interactive dashboard

---

# Key Results

* Extracted rainfall data from **1,000+ rainfall chart images**
* Converted charts into structured time-series dataset
* Trained **XGBoost regression model** for rainfall prediction
* Achieved **~5.32 mm average prediction error**
* Built a deployable inference system using **FastAPI + Streamlit**

---

# System Architecture

```
Rainfall Charts (Images)
        ↓
Computer Vision Extraction (OpenCV + OCR)
        ↓
Structured Rainfall Dataset
        ↓
Feature Engineering
        ↓
XGBoost Forecasting Model
        ↓
FastAPI Prediction API
        ↓
Streamlit Interactive Dashboard
```

---

# Repository Structure

```
cross-modal-rainfall-prediction-system/

app/
│
├── api/           # FastAPI inference service
├── services/      # inference pipeline
├── ui/            # Streamlit dashboard
├── assets/        # dashboard assets
├── database/      # DuckDB queries and schema
└── models/        # trained model artifacts

src/               # core ML modules (feature engineering & extraction)

scripts/
│
├── data/          # dataset building scripts
├── model/         # model training scripts
└── predict/       # inference scripts

data/
├── raw/           # raw rainfall chart images (not included)
└── processed/     # processed datasets

notebooks/         # experimentation notebooks
reports/           # documentation and analysis
tests/             # unit tests

config/            # configuration files

README.md
requirements.txt
```

---

# Machine Learning Pipeline

## 1. Chart Digitization

The dataset consists of rainfall charts rather than tabular data.

A custom computer vision pipeline extracts rainfall information using:

* OpenCV morphological operations
* HIT-MISS transforms
* OCR-based axis detection

This process converts rainfall plots into structured daily rainfall values.

---

## 2. Dataset Construction

The extracted rainfall data is combined with:

* Location metadata
* Temporal features
* External weather variables

to build a machine-learning-ready dataset.

---

## 3. Model Training

Model used:

```
XGBoost Regressor
```

Evaluation metrics:

```
MAE
RMSE
```

Average prediction error:

```
~5.32 mm rainfall
```

---

## 4. Model Serving

The trained model is exposed through:

FastAPI

* REST API for rainfall prediction

Streamlit

* Interactive dashboard for exploring predictions

Users can query rainfall predictions by:

* location
* date

---

# Installation

Clone the repository

```bash
git clone https://github.com/rafli-1729/cross-modal-rainfall-prediction-system.git
cd cross-modal-rainfall-prediction-system
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.\.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Setup

Create a `.env` file in the project root:

```
PYTHONPATH=.
```

This ensures local modules are discoverable during execution.

---

# Running the Application

Start the FastAPI backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Streamlit dashboard

```bash
streamlit run app/ui/app.py
```

Once both services are running:

API available at

```
http://localhost:8000
```

Dashboard available at

```
http://localhost:8501
```

---

# Reproducing the Training Pipeline

If you want to rebuild the model from scratch:

Build dataset

```bash
python -m scripts.data.build_training_dataset
```

Train cross-validation model

```bash
python -m scripts.model.train_cv
```

Train observation model

```bash
python -m scripts.model.train_obs
```

---

# Dataset

The rainfall chart dataset is **not included in this repository** due to size constraints.

Place the dataset in:

```
data/raw/Train
data/raw/Test
```

The extraction pipeline will convert the charts into tabular rainfall data.

---

# Tech Stack

Programming

* Python

Machine Learning

* XGBoost
* Scikit-learn

Computer Vision

* OpenCV
* Tesseract OCR

Data Processing

* Pandas
* NumPy

Serving

* FastAPI
* Streamlit

Database

* DuckDB

---

# Future Improvements

* Cloud deployment
* Automated model retraining
* Real-time weather API integration
* Improved rainfall chart digitization accuracy

---

# Author

Muhammad Rafli Azrarsyah
Actuarial Science — Universitas Gadjah Mada

Interested in:

* Machine Learning
* Data Science
* Forecasting
* Computer Vision