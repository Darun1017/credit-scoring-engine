# IntelliCredit: Alternative Credit Scoring & Portfolio Risk Engine


> **Capstone Project — Business Analytics**  
> An end-to-end data science and machine learning framework designed to evaluate creditworthiness for underserved borrowers using alternative financial and behavioral indicators.

---

## Executive Summary & Problem Statement

Traditional credit scoring models rely heavily on formal bureau history (e.g., CIBIL/FICO scores, legacy banking records), excluding millions of creditworthy thin-file individuals and micro-entrepreneurs. 

**IntelliCredit** bridges this financial inclusion gap by:
1. Synthesizing holistic financial footprints across micro-lending transactions, mobile repayment behavior, and macroeconomic shifts.
2. Developing modular, explainable machine learning scoring models to predict 90-day delinquency risk.
3. Providing clear model interpretability (SHAP values) and risk cut-off analysis to balance approval rates against expected default losses.

---

## Rubric & Academic Policy Compliance

In accordance with course project rubrics:
* **No Pre-Built Datasets Policy:** Pre-built datasets from Kaggle, UCI, or GitHub repositories were strictly prohibited. All data used in this study was programmatically generated using `data_generation.py`.
* **Reproducibility:** A fixed random seed (`seed=42`) is utilized across synthetic data generation and train-test splits.
* **Review 1 Deliverables (20 Marks):** Complete implementation of Data Cleaning, Preprocessing, EDA, Feature Engineering (WOE/IV), Benchmark Modeling, and Model Interpretability.

---

## Repository Structure

```text
credit-scoring-engine/
│
├── README.md                           # Project documentation & overview
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git tracking rules for large data/models
├── data_generation.py                  # Vectorized synthetic dataset generator
│
├── data/
│   ├── data_dictionary.txt             # Variable definitions and schema documentation
│   ├── macroeconomic_indicators.csv    # 36 months of macroeconomic trend indicators
│   ├── borrower_profiles.csv           # Raw profile records (30,000 borrowers)
│   ├── monthly_transactions.csv        # Detailed transactions (360,000 records)
│   └── processed/
│       └── feature_columns.txt         # Final engineered features used for training
│
├── notebooks/                          # Graded Review 1 Deliverables
│   ├── 01_M1_data_cleaning_preprocessing.ipynb    # Data cleaning & type conversion
│   ├── 02_M2_exploratory_data_analysis.ipynb      # Univariate, bivariate & macro EDA
│   ├── 03_M3_feature_engineering.ipynb            # WoE/IV, VIF selection & SMOTE
│   ├── 04_M4_predictive_modeling.ipynb            # Classifiers & hyperparameter tuning
│   └── 05_M5_evaluation_interpretation.ipynb      # ROC, KS-statistic, SHAP & Cut-offs
│
├── src/                                # Modular configuration
│   ├── __init__.py
│   └── config.py                       # Project paths, features & random state
│
├── models/
│   └── model_comparison.csv            # Summary metrics across all trained models
│
└── scripts/
    └── generators/                     # Helper notebook scaffolding scripts
```

---

## Project Workflow & Modules

The project is structured into 5 sequential review modules:

### Module 1: Data Cleaning & Preprocessing (`01_M1`)
* Ingestion of multi-table synthetic lending data (borrower demographics, transaction logs, macro indicators).
* Missing-value treatment demonstrated on 5% injected gaps: KNN imputation (numeric) and mode imputation (categorical).
* Detection of outliers with interquartile range (IQR) bounds and 1st/99th-percentile winsorisation.
* Ordinal encoding (education, location tier) from `src/config.py` and one-hot encoding (gender, platform).

### Module 2: Exploratory Data Analysis (`02_M2`)
* Target imbalance inspection (approx. 92% non-default vs. 8% default rate).
* Cross-sectional distributions across monthly income, debt-to-income ratio (DTI), and credit utilization.
* Correlation matrices and macroeconomic shock impact analysis on repayment rates.

### Module 3: Feature Engineering & Selection (`03_M3`)
* 14 engineered features from transactions and domain ratios; the train/test split is fixed **before** any target-based step, so no test labels leak into features.
* Multicollinearity filtering using **Variance Inflation Factor (VIF)** and a |r| > 0.85 correlation filter.
* Feature selection by **Mutual Information + RFE** on training rows, then a **Weight of Evidence / Information Value** screen that drops features with IV < 0.02 (17 features kept).
* PCA (9 components explain 90% of variance) and t-SNE projections.
* Class imbalance mitigation using **SMOTE** (Synthetic Minority Over-sampling Technique) on training sets only (preventing test set leakage).

### Module 4: Predictive Modeling & Ensembling (`04_M4`)
* Training baseline and tuned configurations for:
  * **Logistic Regression** (traditional regulatory scorecard baseline)
  * **Random Forest** & **Gradient Boosting**
  * **XGBoost** & **LightGBM**
  * **Voting Classifier** & **Stacking Ensemble**
* Grid/Randomized hyperparameter search optimizing for ROC-AUC, with SMOTE applied **inside** each cross-validation fold (imblearn pipeline), so CV scores are honest.

### Module 5: Evaluation & Interpretability (`05_M5`)
* **Performance Metrics:** ROC-AUC, PR-AUC curve, Confusion Matrices, and Kolmogorov-Smirnov (**KS-statistic**).
* **Explainable AI (XAI):** SHAP summary plots, beeswarm distributions, and individual applicant waterfall plots.
* **Risk Policy Cut-offs:** INR cost-optimal threshold (missed default ≈ 65% of the loan vs. lost net interest on a wrongly rejected borrower).

---

## Model Performance Summary (test set, 6,000 borrowers)

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.738** | **0.204** | 0.151 | 68.2% | 0.247 |
| **LightGBM (Tuned)** | 0.735 | 0.201 | 0.162 | 61.4% | 0.256 |
| **Logistic Regression (Tuned)** | 0.728 | 0.197 | 0.148 | 63.8% | 0.241 |
| **Random Forest (Tuned)** | 0.713 | 0.178 | 0.136 | 66.3% | 0.226 |
| *True default probability (ceiling)* | *0.781* | *0.262* | | | |

* Precision/recall/F1 are at the 0.5 threshold. Accuracy is not reported: predicting "no default" for everyone already scores 92%.
* The labels are random draws from a known probability, so no model can exceed AUC 0.781; XGBoost captures about 85% of the achievable discrimination.
* KS = 0.36, top-decile lift 2.9×, and the riskiest 30% of applicants contain 63% of all defaults.
* At the cost-optimal threshold 74.7% of applicants are approved, with a 4.4% default rate (vs. 7.9% if everyone were approved).

---

## Getting Started & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Darun1017/credit-scoring-engine.git
cd credit-scoring-engine
```

### 2. Set Up Python Environment
Ensure Python 3.9+ is installed:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate the Dataset
To regenerate the compliant synthetic dataset:
```bash
python data_generation.py
```
*(Produces `borrower_profiles.csv`, `monthly_transactions.csv`, and `macroeconomic_indicators.csv` inside `data/`)*

### 4. Run the Pipeline
Open Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
Execute the notebooks in sequence inside the `notebooks/` directory:
1. `01_M1_data_cleaning_preprocessing.ipynb`
2. `02_M2_exploratory_data_analysis.ipynb`
3. `03_M3_feature_engineering.ipynb`
4. `04_M4_predictive_modeling.ipynb`
5. `05_M5_evaluation_interpretation.ipynb`

---

---

## Audit Fixes (4 October 2026)

| Issue found | Impact | Fix |
|---|---|---|
| `month_label` built with `timedelta(30 * i)` | repeated months (two "2025-01" rows per borrower) | `pd.date_range(..., freq="MS")`; numeric data unchanged |
| M1 ordinal maps used labels not in the data (`High School`, `Urban`, …) | most rows encoded as −1 | use `src/config.ORDINAL_MAPS`, assert every category maps |
| M3 `platform_risk_premium` / `macro_stress_exposure` were target means over **all** rows | test labels leaked into features | split fixed first; smoothed target encoding on train only; platform dropped (χ² p = 0.71) |
| M3 RFE fitted on one random sample's X with **another** sample's y | RFE selection was noise | single fixed train sample |
| M3 MI / RFE fitted on all rows; constant `total_utility_bills` selected | selection leakage, useless feature | train-only selection, zero-variance filter, WoE/IV screen added (previously claimed but not implemented) |
| M4 CV and tuning on pre-SMOTEd data | RF CV AUC 0.98 vs 0.69 on test | SMOTE inside an `imblearn` pipeline; RF CV now 0.72, consistent with test |
| M5 looked for `test_data.csv` / `*.pkl` | silently re-trained **different, untuned** models | loads M3's `test.csv` and M4's `tuned_*.joblib` |
| Hard-coded `c:/Users/rgdar/...` path; qualitative "Very High" tables; `$` costs | notebook would not run elsewhere; claims contradicted results | relative paths, computed tables, INR costs from the portfolio |
| LightGBM 4.7.0 crashes on Windows | M4 hung | pinned `lightgbm<4.7` |

---

## Individual Contributions (Review 1)

| Member | Contribution | Rubric component (marks) |
|---|---|---|
| **Darun R G** (@Darun1017) | Synthetic data generator, repository setup, initial notebooks M1–M5 and the condensed notebook | Problem & dataset (2) and notebook foundations |
| **goldmauler** (@Goldmauler) | Pipeline audit and correctness fixes (table above), WoE/IV analysis, evidence-based evaluation findings in M5 | Supports Features (2), Modelling (4), Evaluation (3) |
| **Juturu Naga Abhinava Sai** (@abhinav-juturu) | Exploratory data analysis and visualization, Feature engineering or dimensionality reduction | EDA & Visualization (3), Feature Engineering (2) |
| **Adithyaa Seyyone** (@adii11001) | _to be filled by the team_ | |
| **Gowtham Reddy Peddireddy** (@Gowthamreddy285) | Data cleaning and preprocessing | Data cleaning and preprocessing (2)|
