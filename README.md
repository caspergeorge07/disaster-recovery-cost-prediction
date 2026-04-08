# Disaster Recovery Cost Prediction and Resilience Optimization

## Project Overview
This project develops a disaster recovery cost forecasting framework that provides early and accurate projections of disaster recovery expenditures. The system uses FEMA historical disaster records, public assistance obligations, and contextual indicators to predict recovery costs at the point of disaster declaration.

## Business Problem
Government agencies often struggle to estimate recovery costs early, leading to budget misallocation, delayed planning, and emergency borrowing. This project aims to build a predictive framework that improves early-stage financial planning and supports resilience optimization.

## Project Objectives
- Ingest disaster-related data from FEMA Open Data API
- Clean and aggregate recovery cost data
- Explore patterns in disaster recovery spending
- Build predictive models for cost forecasting
- Evaluate and explain model outputs
- Deploy predictions through FastAPI
- Create a Streamlit dashboard for scenario simulation
- Support automation and retraining

## Tech Stack
- Python
- Pandas, NumPy, Scikit-learn, XGBoost
- SHAP
- FastAPI
- Streamlit
- Docker
- GitHub Actions

## Project Structure
```text
data/, notebooks/, src/, api/, dashboard/, models/, reports/, tests/