# Disaster Recovery Cost Prediction and Resilience Optimization

## Project Overview

This project develops an end-to-end machine learning system for predicting disaster recovery costs using FEMA disaster data. The system ingests public FEMA datasets, validates and processes the data, engineers disaster-level features, trains and evaluates machine learning models, exposes predictions through a FastAPI backend, and provides an interactive Streamlit dashboard for scenario analysis and budget comparison.

The project is designed as a portfolio-ready ML engineering system covering:

- Data ingestion
- Data validation
- Feature engineering
- Model training and tuning
- Model evaluation and SHAP explainability
- FastAPI deployment
- Streamlit dashboard
- Docker containerization
- CI/CD automation
- Weekly retraining workflow

---

# Problem Statement

Disaster recovery operations require accurate early-stage financial estimation to support resource allocation, emergency planning, and resilience optimization. However, disaster recovery costs vary significantly depending on disaster type, duration, geography, and historical disaster frequency.

This project uses FEMA disaster datasets and machine learning techniques to estimate disaster recovery costs based on disaster-level characteristics. The system supports scenario simulation, budget comparison, and decision-support workflows.

---

# Objectives

The main objectives of this project are:

- Predict disaster recovery costs using historical FEMA data
- Build an end-to-end reproducible ML pipeline
- Compare multiple regression models
- Provide explainable predictions using SHAP
- Deploy the model through a FastAPI backend
- Create an interactive Streamlit dashboard
- Automate retraining and deployment workflows

---

# System Architecture

```text
                    FEMA Open Data API
                              |
                              v
                    Raw Data Ingestion
                              |
                              v
                    Data Validation Layer
                              |
                              v
                   Feature Engineering Layer
                              |
                              v
               Processed Disaster-Level Dataset
                              |
                              v
                 Model Training & Evaluation
                              |
               --------------------------------
               |                              |
               v                              v
      SHAP Explainability            Best Model Artifact
               |                              |
               --------------------------------
                              |
                              v
                     FastAPI Prediction API
                              |
                              v
                   Streamlit Dashboard UI
                              |
                              v
               Budget Simulation & Forecasting
```

---

# Tech Stack

## Programming Languages

- Python 3.11

## Data Science & ML

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- MLflow

## API & Dashboard

- FastAPI
- Streamlit
- Plotly

## Deployment & Automation

- Docker
- Docker Compose
- GitHub Actions

---

# Project Structure

```text
disaster-recovery-cost-prediction/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
│
├── docs/
│   └── model_card.md
│
├── models/
│   ├── best_model.pkl
│   ├── best_model_metadata.json
│   └── shap_summary.png
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 08_model_training.ipynb
│   ├── 09_model_evaluation.ipynb
│   └── 10_model_tuning.ipynb
│
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── models/
│   └── pipeline/
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── weekly_retrain.yml
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Data Sources

This project uses FEMA Open Data APIs:

- Disaster Declarations Summaries
- Public Assistance Funded Projects
- FEMA Disaster Web Summaries

Data is fetched programmatically and stored locally for validation and processing.

---

# Machine Learning Workflow

## 1. Data Ingestion

The ingestion layer downloads FEMA disaster datasets using API requests and stores them in the raw data folder.

## 2. Data Validation

Validation checks include:

- Missing values
- Duplicate records
- Invalid datatypes
- Schema consistency
- Outlier inspection

## 3. Feature Engineering

Key engineered features include:

- Disaster duration
- Declaration month
- Seasonality
- 5-year historical disaster frequency
- Census region
- High-cost incident flag

The dataset was aggregated to one row per disaster to avoid target leakage and duplicated observations.

---

# Model Training

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The prediction target is:

```text
log_total_obligated_amount
```

The target was log-transformed because disaster recovery cost distributions were highly skewed.

---

# Cross-Validation Strategy

The models were evaluated using:

- 5-fold cross-validation
- R²
- RMSE
- MAE

The final model was selected based on mean cross-validated R² while also considering RMSE and MAE.

---

# Explainability

SHAP analysis was used to interpret model predictions and identify the most influential features.

Key influential features included:

- Declaration year
- Declaration type
- Incident duration
- Historical disaster frequency
- Incident type

The SHAP summary plot is stored at:

```text
models/shap_summary.png
```

---

# API Deployment

The trained model is deployed using FastAPI.

Available endpoints:

| Endpoint | Description |
|---|---|
| `/health` | API health check |
| `/predict-cost` | Predict disaster recovery cost |
| `/model-info` | Return model metadata |

Run locally:

```bash
uvicorn api.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Streamlit Dashboard

The dashboard allows users to:

- Select disaster scenario parameters
- Generate recovery cost predictions
- Compare allocated vs predicted budgets
- View budget gaps
- View SHAP explainability outputs

Run locally:

```bash
streamlit run dashboard/app.py
```

Dashboard URL:

```text
http://127.0.0.1:8501
```

---

# Docker Deployment

## Build and Run

```bash
docker compose up --build
```

## Stop Containers

```bash
docker compose down
```

## Docker Services

| Service | Port |
|---|---|
| FastAPI API | 8000 |
| Streamlit Dashboard | 8501 |

---

# Automated Retraining Pipeline

The project includes an orchestration pipeline that automates:

1. FEMA data ingestion
2. Data validation
3. Feature engineering
4. Model retraining
5. Model comparison
6. Best model replacement

Run manually:

```bash
python -m src.pipeline.run_pipeline
```

---

# CI/CD Workflow

GitHub Actions automates:

- Testing
- Docker image builds
- Weekly retraining workflows

Workflows include:

- `ci.yml`
- `weekly_retrain.yml`

---

# Running Tests

Run all tests:

```bash
python -m pytest
```

Run API tests only:

```bash
python -m pytest tests/test_api.py
```

---

# Installation & Environment Setup

## Create Environment

```bash
conda create -n disaster_recovery python=3.11
conda activate disaster_recovery
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run the Full Project

## Step 1 — Run ingestion

```bash
python scripts/run_ingestion.py
```

## Step 2 — Run validation

```bash
python scripts/run_validation.py
```

## Step 3 — Run feature engineering

```bash
python scripts/run_feature_engineering.py
```

## Step 4 — Train models

```bash
python scripts/run_training.py
```

## Step 5 — Start API

```bash
uvicorn api.main:app --reload
```

## Step 6 — Start dashboard

```bash
streamlit run dashboard/app.py
```

---

# Model Performance Summary

After leakage correction and disaster-level aggregation:

| Model | Test R² |
|---|---|
| Random Forest | ~0.89 |
| Tuned XGBoost | ~0.89 |
| Linear Regression | ~0.64 |

The Random Forest model was selected as the production model because it provided strong predictive performance with stable generalization.

---

# Key Features of the Final System

- Automated FEMA ingestion pipeline
- Leakage-safe ML training
- SHAP explainability
- FastAPI deployment
- Streamlit dashboard
- Dockerized infrastructure
- CI/CD automation
- Weekly retraining support
- End-to-end reproducibility

---

# Known Limitations

- FEMA data may contain incomplete or delayed records
- Extreme disaster events remain difficult to predict
- Predictions should support planning, not replace expert judgement
- Some features may not be available early in a disaster lifecycle
- Geographic and disaster-type imbalance may affect performance

---

# Future Improvements

Potential future enhancements include:

- Cloud deployment (AWS/Azure/GCP)
- Real-time FEMA API streaming
- Time-series forecasting
- Geospatial visualizations
- Advanced ensemble modelling
- Model drift monitoring
- User authentication for dashboard access

---

# Author

**Ndubuaku Casper Ekwueme**

Developed as part of an internship project on:

**Disaster Recovery Cost Prediction and Resilience Optimization**

---

# License

This project is intended for educational, research, and portfolio purposes.