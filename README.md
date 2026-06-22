# Medical Insurance Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A supervised machine learning project that **compares four regression algorithms** to predict annual medical insurance costs based on patient demographics, health conditions, clinical measurements, and insurance policy details.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Algorithms](#algorithms)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Authors](#authors)

---

## Overview

Medical insurance cost prediction is a critical problem in healthcare and actuarial science. This project builds and benchmarks four regression models on a dataset of **100,000 patients** with **52 input features**, aiming to accurately estimate the `annual_medical_cost` for each individual.

The four models are developed in parallel under identical data preprocessing and evaluation conditions, allowing a fair, direct comparison of their predictive capabilities.

**Key goals:**
- Identify which algorithm best predicts medical insurance costs
- Understand which patient features are most influential
- Demonstrate best practices in regression modeling: data leakage prevention, log-transformation of skewed targets, hyperparameter tuning, and cross-validation

---

## Project Structure

```
Medical-Insurance-Price-Prediction/
│
├── Decision tree regressor/
│   └── Decision_Tree_Insurance.ipynb
│
├── Gradient Boosting Regressor/
│   ├── gradient_boosting_regressor_model.ipynb
│   ├── gradient_boosting_regressor_model.py
│   ├── medical_insurance.csv
│   ├── feature_importance_results.csv
│   ├── feature_correlations.csv
│   └── *.png  (training and evaluation visualizations)
│
├── Random Forest Regressor/
│   ├── medical_insurance_random_forest_bagging_colab.ipynb
│   ├── medical_insurance.csv
│   ├── RF_vs_BaggingRegressor_vs_GradientBoosting.md
│   └── *.png  (performance visualizations)
│
├── Support Vector Regression/
│   ├── medical_insurance_analysis.py
│   ├── medical_insurance_svr.py
│   ├── medical_insurance.csv
│   ├── Version_01/  (RBF kernel SVR + saved model artifacts)
│   ├── Version_02/  (SVR iteration with result CSVs)
│   ├── Version_04/  (LinearSVR variant)
│   └── *.png  (residual and prediction plots)
│
└── README.md
```

---

## Dataset

| Property | Details |
|---|---|
| Rows | 100,000 patients |
| Features | 52 input features + 1 target |
| Target Variable | `annual_medical_cost` (USD) |
| File Size | ~21 MB |
| Source | Synthetic dataset for academic use |

### Feature Categories

<details>
<summary><strong>Demographics (8 features)</strong></summary>

| Feature | Description |
|---|---|
| `age` | Age of the individual (18–90 years) |
| `sex` | Gender (Male / Female) |
| `region` | Geographic region |
| `urban_rural` | Urban or rural residence |
| `income` | Annual income (USD) |
| `education` | Education attainment level |
| `marital_status` | Marital status |
| `employment_status` | Employment type/status |

</details>

<details>
<summary><strong>Household (2 features)</strong></summary>

| Feature | Description |
|---|---|
| `household_size` | Number of people in household |
| `dependents` | Number of financial dependents |

</details>

<details>
<summary><strong>Clinical Measurements (5 features)</strong></summary>

| Feature | Description |
|---|---|
| `bmi` | Body Mass Index |
| `systolic_bp` | Systolic blood pressure (mmHg) |
| `diastolic_bp` | Diastolic blood pressure (mmHg) |
| `ldl` | LDL cholesterol level (mg/dL) |
| `hba1c` | Hemoglobin A1c % (blood sugar control) |

</details>

<details>
<summary><strong>Chronic Conditions (11 features)</strong></summary>

| Feature | Description |
|---|---|
| `smoker` | Smoking status (Yes / No) |
| `alcohol_freq` | Frequency of alcohol consumption |
| `chronic_count` | Number of chronic conditions diagnosed |
| `hypertension` | Hypertension diagnosis (0/1) |
| `diabetes` | Diabetes diagnosis (0/1) |
| `asthma` | Asthma diagnosis (0/1) |
| `copd` | COPD diagnosis (0/1) |
| `cardiovascular_disease` | Cardiovascular disease (0/1) |
| `cancer_history` | Cancer history (0/1) |
| `kidney_disease` | Kidney disease (0/1) |
| `liver_disease` | Liver disease (0/1) |
| `arthritis` | Arthritis diagnosis (0/1) |
| `mental_health` | Mental health condition (0/1) |
| `is_high_risk` | High health risk classification (0/1) |

</details>

<details>
<summary><strong>Healthcare Utilization (8 features)</strong></summary>

| Feature | Description |
|---|---|
| `visits_last_year` | Healthcare visits in the past year |
| `hospitalizations_last_3yrs` | Hospitalizations in the past 3 years |
| `days_hospitalized_last_3yrs` | Total hospital days over past 3 years |
| `medication_count` | Number of active prescribed medications |
| `proc_imaging_count` | Imaging procedures (X-ray, MRI, CT) |
| `proc_surgery_count` | Surgical procedures |
| `proc_physio_count` | Physiotherapy / psychology sessions |
| `proc_consult_count` | Specialist consultations |
| `proc_lab_count` | Laboratory diagnostic tests |

</details>

<details>
<summary><strong>Insurance Policy (7 features)</strong></summary>

| Feature | Description |
|---|---|
| `plan_type` | Insurance plan type (HMO, PPO, EPO) |
| `network_tier` | Network tier (Basic, Silver, Gold, Platinum) |
| `deductible` | Annual deductible amount (USD) |
| `copay` | Copayment per service (USD) |
| `policy_term_years` | Length of policy (years) |
| `policy_changes_last_2yrs` | Policy modifications in past 2 years |
| `provider_quality` | Quality rating of primary provider |

</details>

---

## Algorithms

All four models predict the same continuous target (`annual_medical_cost`) and are evaluated under identical conditions.

### 1. Decision Tree Regressor
A single decision tree that recursively splits data by feature thresholds. Serves as the interpretable baseline — fast and explainable, but prone to overfitting without pruning.

### 2. Random Forest Regressor
A bagging ensemble of decision trees with per-split feature randomness. Reduces overfitting vs. a single tree by averaging many uncorrelated trees. Also compared against a pure `BaggingRegressor` to isolate the effect of feature randomness.

### 3. Gradient Boosting Regressor
A sequential boosting ensemble where each new tree corrects the errors of the previous one. Tuned with `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`. Provides built-in feature importance rankings.

### 4. Support Vector Regression (SVR)

Support Vector Regression extends the SVM margin concept to regression by finding a function that stays within an ε-tube around the true values. It is sensitive to feature scale, making `StandardScaler` mandatory, and computationally expensive on large datasets — key design decisions that shaped how this implementation evolved across three versions.

#### Version 1 — RBF Kernel SVR (Optimized for Colab)

Due to SVR's O(n²–n³) training complexity, V1 works on a **15,000-row sample** from the full 100k dataset to keep training feasible on free Colab. Key steps:

- **Feature selection** — `SelectKBest` with `f_regression` scores to retain the top **30 features**, reducing noise and memory usage
- **Dual scaling** — both `X` and the target `y` are independently scaled with `StandardScaler`; the target is inverse-transformed after prediction to recover dollar values
- **Initial training** — `SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')` trained on 10,500 samples in ~2 minutes, yielding **5,531 support vectors**
- **Hyperparameter tuning** — `RandomizedSearchCV` (10 iterations, 3-fold CV) on a 5,000-sample subset with `loguniform` distributions over `C` and `epsilon`; best result: **C≈10.03, ε≈0.081, gamma='scale'**, achieving a best CV R² of **0.7555**
- **Final retraining** — best model retrained on the full 10,500-sample training set (5,799 support vectors)
- **Outputs** — trained model + scaler + selector saved as `.pkl` files; 8 diagnostic plots generated including learning curves and permutation feature importance

**Top features selected (by f_regression):** `risk_score`, `had_major_procedure`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`, `chronic_count`, `is_high_risk`, `smoker`, `hba1c`, `proc_surgery_count`, `medication_count`

#### Version 2 — Iteration with Result Exports

V2 is an intermediate iteration that exports validation and test set predictions to CSV for offline analysis and cross-model comparison. Results are stored in `svr_validation_results.csv` and `svr_test_results.csv`.

#### Version 4 — LinearSVR with Feature Engineering (Full Dataset)

V4 is the most complete version, running on the **full 100,000-row dataset** with stricter leakage controls and richer feature engineering.

**Feature engineering (5 new interaction terms):**

| New Feature | Formula | Rationale |
|---|---|---|
| `age_x_bmi` | `age × bmi` | Captures compound metabolic risk |
| `chronic_x_risk` | `chronic_count × risk_score` | Amplifies risk for multi-condition patients |
| `hosp_x_days` | `hospitalizations × days_hospitalized` | Severity-weighted utilization |
| `proc_total` | Sum of all 5 procedure counts | Overall procedure burden |
| `med_visit_ratio` | `medication_count / (visits + 1)` | Medication intensity per visit |

**Leakage removal** — V4 applies the strictest policy, dropping `annual_premium` (retained in V1) along with all other outcome-derived columns, for a fully "fair" prediction setup.

**Preprocessing pipeline** — `ColumnTransformer` combining:
- `StandardScaler` for numeric and binary features
- `SimpleImputer(strategy='most_frequent') + OneHotEncoder` for categorical features

**Hyperparameter tuning** — `LinearSVR` with a 5-fold cross-validated sweep over `C ∈ {0.01, 0.1, 1, 10, 50, 100, 500}` on the full 70,000-row training set:

| C | CV R² | Notes |
|---|---|---|
| 0.01 | 0.2120 | **Best** |
| 0.1 | 0.2120 | Equal |
| 1 | 0.2120 | Equal |
| 10 | 0.2073 | Slight drop |
| 100 | −0.0937 | Unstable |
| 500 | −0.5473 | Diverged |

Best `C = 0.01` selected. LinearSVR then switched to `SVR(kernel='rbf')` for the final model, trained on a 40,000-sample pool of train+validation data.

**Final test set results:**

| Metric | Value |
|---|---|
| MAE | $1,661.46 |
| RMSE | $3,022.79 |
| R² | 0.0652 |

**Visualizations generated:** CV R² vs C curve, actual vs predicted scatter, residual plot, residual distribution histogram, absolute error by cost range (box plot)

---

## Methodology

### Data Split
All models use the same **70 / 10 / 20** train/validation/test split for a fair comparison.

### Preprocessing Pipeline
1. **Missing values** — `alcohol_freq` (~30% missing) filled with the mode
2. **Encoding** — Categorical features encoded with `LabelEncoder` / `OneHotEncoder` via `ColumnTransformer`
3. **Scaling** — `StandardScaler` applied (critical for SVR)
4. **Target transformation** — `annual_medical_cost` is right-skewed (skewness ≈ 4.03); log-transformed before training and inverse-transformed for evaluation
5. **Leakage removal** — Outcome-derived columns (`monthly_premium`, `total_claims_paid`, `avg_claim_amount`, `claims_count`) dropped to prevent data leakage

### Feature Engineering (SVR)
Interaction terms created to capture non-linear relationships:
- `age × bmi`
- `chronic_count × risk_score`
- `hospitalizations × days_hospitalized`

### Hyperparameter Tuning
`RandomizedSearchCV` with cross-validation used across all models.

### Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error in USD |
| **RMSE** | Root Mean Squared Error — penalises large errors more |
| **R²** | Coefficient of determination — proportion of variance explained |
| **MAPE** | Mean Absolute Percentage Error — relative error |

---

## Results

Each algorithm folder contains:
- Training, validation, and test metrics
- Actual vs. predicted scatter plots
- Residual distribution plots
- Performance heatmaps across train/val/test splits
- Feature importance analysis (tree-based models)

Refer to each subfolder's notebook for the full evaluation output.

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
```

### Running the Notebooks

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Medical-Insurance-Price-Prediction.git
   cd Medical-Insurance-Price-Prediction
   ```

2. Open the notebook for the algorithm you want to explore:
   ```
   Gradient Boosting Regressor/gradient_boosting_regressor_model.ipynb
   Random Forest Regressor/medical_insurance_random_forest_bagging_colab.ipynb
   Support Vector Regression/Version_01/support_vector_regression_V1.ipynb
   Decision tree regressor/Decision_Tree_Insurance.ipynb
   ```

3. The notebooks are also compatible with **Google Colab** — upload the notebook and the `medical_insurance.csv` file to your Drive and run all cells.



