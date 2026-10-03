# 💳 Credit Card Approval Prediction

An end-to-end machine learning project that predicts whether a credit card applicant is likely to become a **bad customer**  someone who defaults or falls significantly behind on repayments  based on their demographic profile, financial situation, and monthly credit history.

The whole workflow lives in a single, well-commented Jupyter notebook: exploratory data analysis (EDA), cleaning, feature engineering, class-imbalance handling, model training/comparison, evaluation, and saving the best model for reuse.

---

## 📌 Problem Statement

Credit card issuers need to decide which applicants are safe to approve. Approving a high-risk applicant leads to financial loss, while rejecting a good applicant loses business. This project frames the task as a **binary classification** problem:

- **`0`** → Good customer (no serious delinquency)
- **`1`** → Bad customer (at least one serious delinquency)

The goal is to learn a model that separates these two groups as well as possible while coping with the **severe class imbalance** in the data.

---

## 📂 Dataset

The project uses two datasets that are joined on the customer `ID`:

| File | Rows | Columns | Description |
|------|-----:|--------:|-------------|
| `application_record.csv` | 438,557 | 18 | Applicant profile: gender, income, education, housing, family, employment, etc. |
| `credit_record.csv` | 1,048,575 | 3 | Monthly credit status per customer (`ID`, `MONTHS_BALANCE`, `STATUS`) |

**Key columns**

- `application_record.csv`: `ID`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`, `OCCUPATION_TYPE`, `CNT_FAM_MEMBERS`
- `credit_record.csv`: `ID`, `MONTHS_BALANCE` (month relative to application), `STATUS`

**`STATUS` values** — `X` (no loan), `C` (closed), `0` (1–29 days past due), `1` (30–59 dpd), `2` (60–89 dpd), `3` (90–119 dpd), `4` (120–149 dpd), `5` (150+ dpd).

**Target definition:** a customer is labelled `1` if they ever had a `STATUS` in `{"2", "3", "4", "5"}` (seriously overdue), otherwise `0` (the per-customer maximum status is taken).

After joining the two files on `ID`, the modelling dataset contains **36,457 customers × 19 columns**, with an imbalanced target: **~98.3% class 0** vs **~1.7% class 1**.

---

## 🗂️ Project Structure

```
Credit-Card-Approval-Prediction-model/
├── application_record.csv              # Applicant profile data (input)
├── credit_record.csv                   # Monthly credit history data (input)
├── notebook_.ipynb                     # End-to-end EDA → modelling notebook
├── credit_default_random_forest.pkl    # Saved Random Forest + SMOTE model (generated)
├── standard_scaler.pkl                 # Saved StandardScaler (generated)
├── .gitignore                          # Ignored Python/ML artifacts
└── README.md
```

> The `.pkl` artifacts are produced by running the notebook and are intentionally excluded from version control (see `.gitignore`).

---

## 🧪 Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected shapes, dtypes, missing values, and duplicated rows.
- Analysed the distribution of `STATUS` and the engineered `TARGET`.
- Visualised the target imbalance, age/income distributions, and default rate by **occupation**, **family status**, and **education level**.

### 2. Data Cleaning
- Replaced the placeholder `DAYS_EMPLOYED = 365243` (invalid) with `NaN`, then imputed it with the median.
- Filled missing `OCCUPATION_TYPE` values with `"Unknown"`.
- Applied **IQR capping** to `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, and `CNT_FAM_MEMBERS` to tame outliers.

### 3. Feature Engineering
- `AGE = -DAYS_BIRTH / 365`
- `EMPLOYMENT_YEARS = -DAYS_EMPLOYED / 365`
- `INCOME_PER_FAMILY_MEMBER = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
- `HAS_CHILDREN = CNT_CHILDREN > 0`
- `IS_WORKING = EMPLOYMENT_YEARS > 0`
- Dropped the raw `DAYS_BIRTH` / `DAYS_EMPLOYED` columns.

### 4. Encoding & Scaling
- **One-hot encoding** for categorical features (`drop_first=True`) → 52 columns.
- **`StandardScaler`** applied to the numerical features (the fitted scaler is reused to transform new inputs).

### 5. Train/Test Split
- 80/20 split with `stratify=y` and `random_state=42` to preserve the class ratio.
- The test set is left **untouched** by resampling so it reflects the real-world distribution.

### 6. Handling Class Imbalance
- **`class_weight="balanced"`** for Logistic Regression and Random Forest.
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied **only to the training set** to synthesise minority-class samples (493 → 28,672).

---

## 🤖 Models & Results

Four models were trained and evaluated on the same held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.5876 | 0.0209 | 0.5122 | 0.0402 | 0.5576 |
| Random Forest (`class_weight="balanced"`) | 0.9738 | 0.2463 | 0.2683 | 0.2568 | **0.8263** |
| **Random Forest + SMOTE** ✅ | 0.9765 | 0.2391 | 0.1789 | 0.2047 | 0.7957 |
| XGBoost + SMOTE | 0.9768 | 0.2386 | 0.1707 | 0.1991 | 0.6531 |

**Selected final model:** **Random Forest + SMOTE** (200 trees) — persisted together with the fitted `StandardScaler`. It achieves ~**97.65% test accuracy** and the strongest balance of accuracy and ROC-AUC among the top performers.

Because the positive class is extremely rare, **accuracy alone is misleading**; `ROC-AUC`, precision, recall, and F1 are reported alongside it.

### Analysis artefacts
- Confusion matrix for the final model.
- ROC curve for the final model.
- Top-15 feature importances from the Random Forest.

---

## 🛠️ Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.14 | Language runtime |
| pandas | 3.0 | Data loading & wrangling |
| NumPy | 2.5 | Numerical operations |
| scikit-learn | 1.9 | Preprocessing, models, metrics |
| imbalanced-learn | 0.14 | SMOTE over-sampling |
| XGBoost | 3.4 | Gradient-boosted baseline |
| Matplotlib / Seaborn | 3.11 / 0.13 | Visualisation |
| joblib | 1.6 | Model persistence |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/PIYUSH-NEXTGEN/Credit-Card-Approval-Prediction-model.git
cd Credit-Card-Approval-Prediction-model
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib jupyter
```

### 4. Run the notebook
```bash
jupyter notebook notebook_.ipynb
```
Run the cells top-to-bottom. The notebook reads the two CSV files from the project root, performs the full pipeline, and writes `credit_default_random_forest.pkl` and `standard_scaler.pkl` at the end.

---

## 📦 Using the Saved Model

```python
import joblib
import pandas as pd

# Load the persisted model and scaler
model = joblib.load("credit_default_random_forest.pkl")
scaler = joblib.load("standard_scaler.pkl")

# `sample` must be preprocessed exactly as in the notebook:
#   same cleaning, feature engineering, one-hot encoding (same columns/order),
#   and numerical columns scaled with `scaler`.
# probabilities = model.predict_proba(scaler.transform(sample[scaler.feature_names_in_]))[:, 1]
# prediction   = model.predict(scaler.transform(sample[scaler.feature_names_in_]))
```

---



## 🙌 Acknowledgements

- Kaggle for the *Credit Card Approval Prediction* dataset.
