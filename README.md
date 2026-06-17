# Kenya Financial Status Prediction — SDC DataSprint 2026

**Strathmore Data Community (SDC) × iLab Africa | June 2026**

---

## Project Overview

A multiclass classification solution predicting whether a Kenyan adult's financial
situation has **Improved**, **Stayed the Same**, or **Worsened**, built on the
2024 FinAccess Household Survey — the most comprehensive financial inclusion dataset
published for Kenya, covering 20,871 respondents across all 47 counties.

**Best result:** Random Forest — Weighted F1 = **0.536** on held-out test set.

---

## Problem Statement

> Which factors most strongly predict financial deterioration among Kenyan adults,
> and what should policymakers, banks, NGOs, or development partners prioritise?

52.6% of 20,871 Kenyans surveyed reported worsened financial status.
43.5% experienced a financial shock in the past year.
9.9% remain fully excluded from all financial services.
This is a real-world classification problem with genuine policy stakes.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | 2024 FinAccess Household Survey (Central Bank of Kenya, KNBS, FSD Kenya) |
| Rows | 20,871 respondents |
| Columns | 28 features |
| Target | financial_status: Worsened 52.6% / Stayed the same 26.9% / Improved 20.5% |
| Geography | All 47 Kenyan counties |
| Kaggle | https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024 |

**Feature categories:** Demographics · Livelihood & Income · Mobile & Digital Access
· Financial Behaviour · Financial Health & Literacy

---

---

## Methodology

### 1. Data Cleaning (`wrangle()`)

All cleaning is encapsulated in a single reproducible `wrangle()` function:

| Step | Action |
|------|--------|
| Missing values | `barriers_bank` (27.5% null) → imputed with `"No barrier"` |
| Education level | Stripped whitespace; merged "Refused to Answer", "Other (Specify)", "95" → `"Other"` |
| Marital status | Merged "Don't know" and "Refused to Answer" → `"Other"` |
| All nulls | Remaining nulls filled via `.fillna("No barrier")` as first step |
| Output | 20,871 × 28, zero nulls |

### 2. Exploratory Data Analysis

| Chart | Key Finding |
|-------|-------------|
| Target distribution | 52.6% Worsened — class imbalance confirmed |
| Status by Shock | Shocked: 59.6% worsened vs Not shocked: 47.3% — **12.3pp gap** |
| Status by Education | No education: 57.3% worsened; University: 38.6% worsened — **19pp gap** |
| Status by Income | <KES 2.5K: 57.3% worsened; KES 10K+: 42.6% worsened — **15pp gap** |
| Status by Location | Rural: 53.6% worsened; Urban: 50.8% worsened — moderate gap |

### 3. Feature Engineering

- **Binary encoding:** Yes/No, Urban/Rural, Male/Female, Usage/Non-usage columns → 0/1
- **Income grouping:** `pd.cut` into <2.5K, 2.5K–5K, 5K–10K, 10K+ buckets
- **Multicollinearity check:** VIF analysis revealed `mobile_money_access` (VIF=66)
  and `formal_service_use` (VIF=65) as highly collinear. Retained — tree models
  are robust to multicollinearity.

### 4. Model Development

All models wrapped in `sklearn.Pipeline`:
`OneHotEncoder (category_encoders) → Classifier`

| Model | Val Weighted F1 | Test Weighted F1 | Training Time |
|-------|-----------------|------------------|---------------|
| Random Forest (baseline) | 0.543 | **0.536** ✅ SELECTED | 0.95s |
| XGBoost (baseline) | 0.539 |  | 2.91s |
| LightGBM (baseline) | 0.523 |
| RF + Optuna (30 trials) | 0.533 | — | — |


**Model selection rationale:** Random Forest achieves the highest validation Weighted F1
(0.543) and demonstrates stable, consistent results across all classes. Optuna tuning
with 30 trials did not improve on the baseline, suggesting the model had reached a
performance plateau within the current feature set.

**Class imbalance handling:** `class_weight="balanced"` for RF and LGBM;
`compute_sample_weight("balanced")` for XGBoost.
**Split:** 60% train / 20% val / 20% test — stratified by `financial_status`.

### 5. Evaluation

Primary metric: **Weighted F1-Score** (accounts for class imbalance).
```
Random Forest — Final Test Set Report:
                 precision    recall  f1-score   support

       Improved      0.391     0.410     0.400       856
Stayed the same      0.414     0.332     0.369      1122
       Worsened      0.648     0.701     0.674      2197

       accuracy                          0.543      4175
      macro avg      0.485     0.481     0.481      4175
   weighted avg      0.533     0.543     0.536      4175
```


---

## Key Findings

### From the Model (RF Feature Importance)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | monthly_income | 7.18% | Primary resilience predictor — income buffers against all shocks |
| 2 | household_size | 6.20% | Larger households dilute per-capita resources — amplified vulnerability |
| 3 | prodsum1 | 5.49% | Financial product breadth = inclusion depth = resilience |
| 4 | Sex | 2.13% | Gender-based structural vulnerability captured by model |
| 5 | nfhi_11 (food security) | 2.04% | Food insecurity is a leading indicator of financial decline |
| 6 | location_type | 1.96% | Rural slightly more vulnerable; urban not protected |
| 7 | experienced_shock | 1.93% | 12.3pp bivariate gap — effect distributed across downstream features |

### From EDA (Bivariate Analysis)

- **Shocks:** 59.6% vs 47.3% worsened rate — sharpest single split in the dataset
- **Education:** 19pp resilience gap between no education and university
- **Income:** 15pp gap between lowest and highest income quartiles
- **Mobile/formal collinearity:** VIF ~65 — they measure the same inclusion channel

---

## Recommendations

| Stakeholder | Recommendation | Evidence Base |
|-------------|----------------|---------------|
| **Government** | Expand shock-responsive social protection (drought insurance, | experienced_shock: 12.3pp gap |
|  | emergency cash transfers). 43.5% of population is shock-exposed. | |
| **Banks & SACCOs** | Design products for sub-KES 5,000/month earners: | monthly_income #1 driver; |
|  | micro-savings, salary-advance, flexible repayment. | income gradient across all quartiles |
| **Development Agencies** | Prioritise per-capita household targeting; large | household_size #2 driver |
|  | households need scaled, not flat, support. | |
| **NGOs** | Combine financial literacy with income support + | fl_score contributes; literacy alone |
|  | shock mitigation. Standalone literacy training insufficient. | insufficient without income buffer |
| **Mobile Operators** | Deepen financial product bundling; move customers | prodsum1 #3 driver; each product |
|  | from 1–2 to 4–5 products for resilience uplift. | adds a resilience layer |
| **All Stakeholders** | Gender-responsive programming is essential. | Sex ranks #4 — 2.13% importance |
|  | Women face structural financial vulnerability. | |

---

## Technologies

```
Python 3.10+         pandas, numpy, matplotlib, seaborn, re, os
scikit-learn         Pipeline, RandomForestClassifier, train_test_split,
                     ConfusionMatrixDisplay, LabelEncoder, 
XGBoost              XGBClassifier with sample_weight + FunctionTransformer
LightGBM             LGBMClassifier
category_encoders    OneHotEncoder (pipeline-compatible, use_cat_names=True)
Optuna               Hyperparameter optimisation (30 trials per model)
statsmodels          variance_inflation_factor (VIF analysis)
pickle               Model serialisation (rf_model.pkl)
```

---

## Installation

```bash
git clone https://github.com/anyangamorris/DataSprint2026
cd DataSprint2026

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm \
            category_encoders optuna statsmodels openpyxl

```

---

## Reproducibility

1. Place `finaccess2024_datasprint.csv` (or `.xlsx`) in the project root
2. Update `data_path` in Cell 3 if filename differs
3. Run: **Kernel → Restart & Run All**
4. `random_state=42` is set for all models — results are deterministic
5. Trained model saved to `./models/rf_model.pkl`

---

## Team

| Member | Role | Contributions |
|--------|------|---------------|
| Morris Anyanga | ML Engineer | Data wrangling, EDA, feature engineering, model development, Optuna tuning, feature importance, confusion matrix |
| Earlyn Sonne | Designer | PowerPoint deck, chart design, slide layout and visual storytelling |
| Rocky Njuguna | Analyst & PM | Project coordination, narrative writing, policy interpretation, README |



---

## Future Improvements

- [ ] Retrain final model on train + val combined (est. +0.01–0.03 Weighted F1)
- [ ] Address "Stayed the same" recall (0.309) — SMOTE or custom class weights
- [ ] Add SHAP beeswarm plot for global feature importance explanation
- [ ] CatBoost evaluation (native categorical handling — may outperform OHE+RF)
- [ ] County-level choropleth map of financial status distribution
- [ ] Streamlit inference demo using saved rf_model.pkl
- [ ] Cross-validated Optuna (3-fold CV in objective) for more reliable tuning

---

*Strathmore Data Community × iLab Africa*
