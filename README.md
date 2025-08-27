# Credit Risk Assessment — Give Me Some Credit (GMSC)

Predict **probability of default (PD)** from consumer **financial history** and turn scores into **actionable decisions**.  
Leakage-safe scikit-learn pipeline, model benchmarks, **precision/recall + ROC-AUC**, **cost-based thresholding**, and **decile lift/gain** analysis.

> **Highlights**
> - Best **ROC-AUC: 0.868** · **PR-AUC: 0.408** (base default rate **6.68%**)
> - **F1** operating point: **thr 0.20 → Precision 0.401 / Recall 0.502**
> - **Cost-optimal** point (C_FN » C_FP): **thr 0.08 → Precision 0.245 / Recall 0.737**
> - **KS: 0.574** · **Top 10% captures 55.2%** of defaults (**73.5% in top 20%**)

---

## 📦 Project Structure

```
.
├─ data/
│  ├─ cs-training.csv
│  ├─ cs-test.csv
│  └─ sampleEntry.csv          # optional (Kaggle template)
├─ notebooks/
│  ├─ Credit_Risk_Assessment_GMSC.ipynb
│  └─ Credit_Risk_Assessment_GMSC_TUNED.ipynb
├─ models/                     # saved pipelines (.joblib)
├─ outputs/                    # predictions, decile tables, charts
└─ README.md
```

> The “TUNED” notebook adds `RandomizedSearchCV` on a stratified subsample + full-train refit.

---

## 🗂 Dataset

**Give Me Some Credit (GMSC)** — consumer credit features with two-year default label.  
- `cs-training.csv` — labeled training set with target **`SeriousDlqin2yrs`** (1 = default).  
- `cs-test.csv` — unlabeled test set for out-of-sample scoring.  
- `sampleEntry.csv` — optional Kaggle submission template.

Place CSVs under `data/`. No network calls required.

---

## ⚙️ Environment

- Python 3.9+
- Install deps:

```
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
matplotlib
scipy
joblib
jupyter
```

---

## 🚀 Quickstart

1) Clone and add data:
```
git clone <your-repo-url>
cd <your-repo>
mkdir -p data outputs models notebooks
# copy cs-training.csv / cs-test.csv / sampleEntry.csv into ./data
```

2) Launch Jupyter and open a notebook:
```
jupyter lab
# open notebooks/Credit_Risk_Assessment_GMSC.ipynb
# or   notebooks/Credit_Risk_Assessment_GMSC_TUNED.ipynb
```

3) Run all cells. Artifacts will be saved to `outputs/` (CSV, charts) and `models/` (joblib).

---

## 🔬 Modeling Approach

### Preprocessing (leak-safe, inside the Pipeline)
- **Median imputation** for numerics **+ missing-value indicators**
- **Quantile clipping** at **1st/99th percentiles** to reduce outlier leverage
- **Standardization** (`StandardScaler`)
- Implemented via `ColumnTransformer` → `Pipeline` so transforms are **fit on train only**

### Models
- **LogisticRegression** (`class_weight="balanced"`)
- **RandomForestClassifier**
- **GradientBoostingClassifier** (best on this tabular data)

### Tuning (tuned notebook)
- `RandomizedSearchCV`, **CV=3**, **n_iter ≈ 25** on **20k** stratified subsample
- Refit best params on **full training set**; evaluate on held-out validation

---

## 📊 Evaluation

### Global metrics
- **ROC-AUC:** **0.868** (GB best) · **0.861** (LR) · **0.849** (RF)
- **PR-AUC (Average Precision):** **0.408**
- **Base default rate:** **6.68%**

### Thresholded operating points
- **F1-optimal:** threshold **0.20** → **Precision 0.401 / Recall 0.502**
- **Cost-optimal:** threshold **0.08** (with **C_FN ≫ C_FP**) → **Precision 0.245 / Recall 0.737**

### Portfolio / ranking analytics
- **Deciles:** sort by PD, split into 10 buckets  
  - **Decile 1** event rate **36.9%**, **Lift 5.52×** over baseline  
  - **Top 10%** captures **55.2%** of all defaults; **Top 20%** captures **73.5%**
- **KS (max):** **0.574** · **Gini:** ~**0.736** (≈ 2·AUC − 1)

**Figures (rendered in notebooks):**
- ROC curve, PR curve  
- Confusion matrices @ F1-optimal & cost-optimal thresholds  
- Expected **Cost vs Threshold** curve  
- **Cumulative Gain** & **Decile Lift** charts  
- Decile table export: `outputs/credit_decile_table.csv`

---

## 💸 Cost-Based Thresholding (edit this cell in the notebook)

```
C_FN = 10.0   # cost of a missed defaulter (false negative)
C_FP = 1.0    # cost of reviewing a non-defaulter (false positive)
```

The code sweeps thresholds, computes expected cost, highlights the **min-cost threshold**, and prints precision/recall + confusion matrix at that operating point.

---

## ♻️ Reproducibility & Artifacts

- Fixed `random_state` for splits/models  
- End-to-end `Pipeline` saved with **`joblib`** → `models/`  
- Predictions and decile table exported to **CSV** → `outputs/`  
- Notebooks include code comments and theory markups

**Artifacts (examples):**
- `models/credit_risk_<best_model>.joblib`  
- `models/credit_risk_tuned_gb.joblib`  
- `outputs/credit_risk_submission.csv`  
- `outputs/credit_risk_submission_tuned_gb.csv`  
- `outputs/credit_decile_table.csv`

---

## 🏷️ License & Attribution

- Data: **Give Me Some Credit** (Kaggle)—use per dataset terms.  

## 🙌 Acknowledgments

- scikit-learn (pipelines & models)  
- Matplotlib (visualizations)  
- GMSC community for benchmarking ideas
# Credit Risk Assessment — Give Me Some Credit (GMSC)

Predict **probability of default (PD)** from consumer **financial history** and turn scores into **actionable decisions**.  
Leakage-safe scikit-learn pipeline, model benchmarks, **precision/recall + ROC-AUC**, **cost-based thresholding**, and **decile lift/gain** analysis.

> **Highlights**
> - Best **ROC-AUC: 0.868** · **PR-AUC: 0.408** (base default rate **6.68%**)
> - **F1** operating point: **thr 0.20 → Precision 0.401 / Recall 0.502**
> - **Cost-optimal** point (C_FN » C_FP): **thr 0.08 → Precision 0.245 / Recall 0.737**
> - **KS: 0.574** · **Top 10% captures 55.2%** of defaults (**73.5% in top 20%**)

---

## 📦 Project Structure

```
.
├─ data/
│  ├─ cs-training.csv
│  ├─ cs-test.csv
│  └─ sampleEntry.csv          # optional (Kaggle template)
├─ notebooks/
│  ├─ Credit_Risk_Assessment_GMSC.ipynb
│  └─ Credit_Risk_Assessment_GMSC_TUNED.ipynb
├─ models/                     # saved pipelines (.joblib)
├─ outputs/                    # predictions, decile tables, charts
└─ README.md
```

> The “TUNED” notebook adds `RandomizedSearchCV` on a stratified subsample + full-train refit.

---

## 🗂 Dataset

**Give Me Some Credit (GMSC)** — consumer credit features with two-year default label.  
- `cs-training.csv` — labeled training set with target **`SeriousDlqin2yrs`** (1 = default).  
- `cs-test.csv` — unlabeled test set for out-of-sample scoring.  
- `sampleEntry.csv` — optional Kaggle submission template.

Place CSVs under `data/`. No network calls required.

---

## ⚙️ Environment

- Python 3.9+
- Install deps:

```
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
matplotlib
scipy
joblib
jupyter
```

---

## 🚀 Quickstart

1) Clone and add data:
```
git clone <your-repo-url>
cd <your-repo>
mkdir -p data outputs models notebooks
# copy cs-training.csv / cs-test.csv / sampleEntry.csv into ./data
```

2) Launch Jupyter and open a notebook:
```
jupyter lab
# open notebooks/Credit_Risk_Assessment_GMSC.ipynb
# or   notebooks/Credit_Risk_Assessment_GMSC_TUNED.ipynb
```

3) Run all cells. Artifacts will be saved to `outputs/` (CSV, charts) and `models/` (joblib).

---

## 🔬 Modeling Approach

### Preprocessing (leak-safe, inside the Pipeline)
- **Median imputation** for numerics **+ missing-value indicators**
- **Quantile clipping** at **1st/99th percentiles** to reduce outlier leverage
- **Standardization** (`StandardScaler`)
- Implemented via `ColumnTransformer` → `Pipeline` so transforms are **fit on train only**

### Models
- **LogisticRegression** (`class_weight="balanced"`)
- **RandomForestClassifier**
- **GradientBoostingClassifier** (best on this tabular data)

### Tuning (tuned notebook)
- `RandomizedSearchCV`, **CV=3**, **n_iter ≈ 25** on **20k** stratified subsample
- Refit best params on **full training set**; evaluate on held-out validation

---

## 📊 Evaluation

### Global metrics
- **ROC-AUC:** **0.868** (GB best) · **0.861** (LR) · **0.849** (RF)
- **PR-AUC (Average Precision):** **0.408**
- **Base default rate:** **6.68%**

### Thresholded operating points
- **F1-optimal:** threshold **0.20** → **Precision 0.401 / Recall 0.502**
- **Cost-optimal:** threshold **0.08** (with **C_FN ≫ C_FP**) → **Precision 0.245 / Recall 0.737**

### Portfolio / ranking analytics
- **Deciles:** sort by PD, split into 10 buckets  
  - **Decile 1** event rate **36.9%**, **Lift 5.52×** over baseline  
  - **Top 10%** captures **55.2%** of all defaults; **Top 20%** captures **73.5%**
- **KS (max):** **0.574** · **Gini:** ~**0.736** (≈ 2·AUC − 1)

**Figures (rendered in notebooks):**
- ROC curve, PR curve  
- Confusion matrices @ F1-optimal & cost-optimal thresholds  
- Expected **Cost vs Threshold** curve  
- **Cumulative Gain** & **Decile Lift** charts  
- Decile table export: `outputs/credit_decile_table.csv`

---

## 💸 Cost-Based Thresholding (edit this cell in the notebook)

```
C_FN = 10.0   # cost of a missed defaulter (false negative)
C_FP = 1.0    # cost of reviewing a non-defaulter (false positive)
```

The code sweeps thresholds, computes expected cost, highlights the **min-cost threshold**, and prints precision/recall + confusion matrix at that operating point.

---

## ♻️ Reproducibility & Artifacts

- Fixed `random_state` for splits/models  
- End-to-end `Pipeline` saved with **`joblib`** → `models/`  
- Predictions and decile table exported to **CSV** → `outputs/`  
- Notebooks include code comments and theory markups

**Artifacts (examples):**
- `models/credit_risk_<best_model>.joblib`  
- `models/credit_risk_tuned_gb.joblib`  
- `outputs/credit_risk_submission.csv`  
- `outputs/credit_risk_submission_tuned_gb.csv`  
- `outputs/credit_decile_table.csv`

---

## 🏷️ License & Attribution

- Data: **Give Me Some Credit** (Kaggle)—use per dataset terms.  
- Code: MIT (add a `LICENSE` file if desired).

## 🙌 Acknowledgments

- scikit-learn (pipelines & models)  
- Matplotlib (visualizations)  
- GMSC community for benchmarking ideas
