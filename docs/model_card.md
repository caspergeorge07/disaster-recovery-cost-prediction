# Model Card — Disaster Recovery Cost Prediction Model

---

# Model Overview

## Model Name

Disaster Recovery Cost Prediction Model

## Model Type

Supervised Regression Model

## Final Selected Model

Random Forest Regressor

## Frameworks Used

- Scikit-learn
- XGBoost
- Pandas
- NumPy
- SHAP

---

# Model Purpose

The purpose of this model is to estimate disaster recovery costs using FEMA disaster-related features.

The model supports:

- Early-stage disaster budgeting
- Recovery planning
- Scenario simulation
- Resource allocation analysis
- Resilience optimization workflows

This system is intended as a decision-support tool and not a replacement for official FEMA financial assessment procedures.

---

# Prediction Target

The model predicts:

```text
log_total_obligated_amount
```

This is the log-transformed disaster recovery cost.

Predictions are converted back into USD using:

```python
np.expm1(prediction)
```

---

# Intended Users

Potential users include:

- Emergency planning analysts
- Disaster response teams
- Resilience planning teams
- Public sector analysts
- Policy researchers
- Financial planning teams

---

# Training Data

## Data Sources

The model was trained using FEMA Open Data datasets:

- Disaster Declarations Summaries
- Public Assistance Funded Projects
- FEMA Disaster Web Summaries

## Aggregation Strategy

The dataset was aggregated to one row per disaster event to avoid:

- duplicated targets
- target leakage
- over-optimistic validation performance

## Dataset Size

Final disaster-level dataset:

```text
~5,163 disaster records
```

---

# Features Used

## Numeric Features

- declaration_year
- declaration_month
- incident_duration_days
- state_5yr_disaster_count
- high_cost_incident
- fyDeclared
- tribalRequest
- fipsStateCode
- fipsCountyCode
- placeCode
- region

## Categorical Features

- state
- incidentType
- declarationType
- season
- census_region

---

# Features Excluded

The following features were excluded due to leakage risk:

- total_obligated_amount
- log_total_obligated_amount
- project_count
- avg_project_amount
- hmProgramDeclared
- iaProgramDeclared
- paProgramDeclared
- ipProgramDeclared

Identifier-style columns were also excluded.

---

# Preprocessing Pipeline

The modelling pipeline uses:

## Numeric Features

- StandardScaler

## Categorical Features

- OneHotEncoder(handle_unknown="ignore")

A ColumnTransformer combines preprocessing steps within the sklearn pipeline.

---

# Models Evaluated

The following models were compared:

| Model | Purpose |
|---|---|
| Linear Regression | Baseline benchmark |
| Random Forest Regressor | Nonlinear ensemble model |
| XGBoost Regressor | Gradient boosting model |

---

# Evaluation Strategy

The models were evaluated using:

- 5-fold cross-validation
- Held-out test set evaluation

Metrics used:

- R²
- RMSE
- MAE

---

# Final Performance

## Held-Out Test Performance

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Random Forest | ~0.89 | ~2.92 | ~1.20 |
| Tuned XGBoost | ~0.89 | ~2.96 | ~1.37 |
| Linear Regression | ~0.64 | ~5.38 | ~4.05 |

The Random Forest model was selected because it provided strong predictive performance with stable generalization.

---

# Performance by Incident Type

The model generally performed better on:

- Fire incidents
- Floods
- Snowstorms

Higher errors were observed for:

- Hurricanes
- Severe storms
- Biological incidents

These incident types showed higher variability in disaster recovery costs.

---

# Performance by Region

Performance varied across Census regions.

The model performed more consistently in regions with:

- larger sample sizes
- more stable historical disaster patterns

Smaller or more heterogeneous regions showed slightly higher error variation.

---

# Explainability

SHAP explainability was used to understand model behaviour.

Key influential features included:

- declaration year
- declaration type
- incident duration
- historical disaster frequency
- incident type

The SHAP summary plot is stored at:

```text
models/shap_summary.png
```

---

# API Integration

The trained model is served through FastAPI.

Available endpoints:

| Endpoint | Description |
|---|---|
| `/health` | API health check |
| `/predict-cost` | Generate cost prediction |
| `/model-info` | Return model metadata |

---

# Dashboard Integration

The Streamlit dashboard allows users to:

- simulate disaster scenarios
- compare allocated vs predicted budgets
- view budget gaps
- explore SHAP explainability outputs

---

# Ethical Considerations

This model should support human decision-making and not replace expert judgement.

Important considerations include:

- uncertainty in disaster outcomes
- uneven regional disaster representation
- potential historical bias in FEMA data
- variability in disaster severity

Predictions should always be interpreted carefully.

---

# Known Limitations

## Data Limitations

- FEMA records may contain missing or delayed information
- Some disasters contain limited observations
- Disaster costs are highly variable and difficult to model perfectly

## Modelling Limitations

- Extreme high-cost disasters remain difficult to predict accurately
- Rare incident types may generalize less reliably
- Some useful real-world variables may not exist in FEMA datasets

## Operational Limitations

- Predictions depend on available input features
- The system is designed for planning support, not legal or financial approval

---

# Recommended Use Cases

Recommended uses include:

- Disaster budgeting
- Scenario simulation
- Recovery planning
- Resilience analysis
- Educational demonstrations
- ML engineering portfolio projects

---

# Not Recommended For

This system should not be used for:

- Final government financial approval
- Legal compliance decisions
- Replacing official FEMA review processes
- Fully automated disaster funding decisions

---

# Monitoring & Maintenance

The project includes:

- automated retraining pipeline
- CI/CD workflows
- Docker deployment
- weekly GitHub Actions retraining support

The model should be retrained periodically as new FEMA data becomes available.

---

# Reproducibility

The project supports reproducibility through:

- sklearn Pipelines
- MLflow experiment tracking
- Docker containers
- GitHub Actions workflows
- version-controlled training pipelines

---

# Future Improvements

Potential future enhancements include:

- geospatial modelling
- cloud deployment
- drift detection
- time-series forecasting
- advanced ensemble models
- interactive SHAP dashboards

---

# Author

Developed as part of an internship project:

**Disaster Recovery Cost Prediction and Resilience Optimization**