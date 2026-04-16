# W8 · Tuesday — Neural Networks + Data Cleaning
### PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar

---

## Overview

This assignment combines two parallel tracks:

1. **Data Cleaning** — Audit and repair a messy hospital admissions dataset (`hospital_records.csv`) that mirrors real-world data quality problems encountered in clinical settings.
2. **Neural Network from Scratch** — Build a 3-layer neural network in pure NumPy (no PyTorch, no TensorFlow) to predict 30-day hospital readmissions — one of the most studied problems in clinical ML.

The seven sub-steps are designed to flow sequentially: cleaning decisions in Sub-steps 1–2 directly affect model performance in Sub-steps 3–5.

---

## Repository Structure

```
week-08/
└── tuesday/
    ├── W8_Tuesday_Assignment.ipynb   # Main notebook — all 7 sub-steps
    ├── generate_dataset.py           # Generates synthetic hospital_records.csv
    ├── hospital_records.csv          # Dataset (run generate_dataset.py if absent)
    ├── README.md                     # This file
    └── (figures saved on run)
        ├── cleaning_distributions.png
        ├── training_loss.png
        ├── model_evaluation.png
        ├── cost_threshold_curve.png
        ├── accuracy_trap.png
        └── embedding_tsne.png
```

---

## Requirements

### Python Version
```
Python >= 3.9
```

### Dependencies
Install all dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Full list:

| Package | Version tested | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Neural network implementation (no PyTorch/TF) |
| `pandas` | ≥ 1.5 | Data loading and cleaning |
| `matplotlib` | ≥ 3.6 | Training curves, confusion matrices, t-SNE plots |
| `scikit-learn` | ≥ 1.2 | Baseline model, StandardScaler, metrics |
| `jupyter` | ≥ 1.0 | Running the notebook |

> **Note:** `torch`, `tensorflow`, and `keras` are intentionally **not** used. The assignment mandates a pure NumPy implementation for Sub-steps 3–5.

---

## How to Run

### Step 1 — Clone / set up
```bash
cd week-08/tuesday/
```

### Step 2 — Generate dataset (if `hospital_records.csv` is absent)
```bash
python generate_dataset.py
```
This creates `hospital_records.csv` with **2,040 rows** (2,000 base + ~2% synthetic duplicates), simulating the messy real-world file from the LMS. Intentional issues include:
- String-noise in age (`"45 yrs"`)
- Impossible ages (negative, > 110)
- Zero and extreme-outlier BMI values
- Mixed gender labels (`M / Male / male / 1 / F / Female / female / 0`)
- Compound blood-pressure column (`"120/80"`)
- Categorical strings in glucose (`"high"`, `"low"`)
- Sign-flip errors in creatinine
- ~6% scattered missing values across all columns
- ~6% positive rate in the target (class imbalance)

> If the LMS-provided `hospital_records.csv` is available, drop it into this folder and skip Step 2 — the notebook reads from `hospital_records.csv` directly.

### Step 3 — Launch notebook
```bash
jupyter notebook W8_Tuesday_Assignment.ipynb
```
Or in JupyterLab:
```bash
jupyter lab W8_Tuesday_Assignment.ipynb
```

### Step 4 — Run all cells
`Kernel → Restart & Run All`

Expected runtime: **2–4 minutes** (dominated by 2000-epoch NN training and t-SNE in Sub-step 7).

---

## Sub-step Summary & Approach

### 🟢 Sub-step 1 — Data Quality Audit
Eight dedicated audit functions systematically inspect every column:
- Missing value census (counts + percentages)
- Duplicate row detection
- Age: non-numeric strings, negatives, biologically impossible values
- BMI: zeros (missing markers), physiological outliers
- Blood pressure: compound format parsing
- Gender: value-count enumeration of inconsistent labels
- Glucose: non-numeric label detection
- Creatinine: negative value detection

**Output:** A full issues table with planned fix for each problem.

---

### 🟢 Sub-step 2 — Data Cleaning
Each column gets a dedicated cleaning function following explicit rules:

| Column | Strategy |
|---|---|
| `age` | Strip "yrs" noise → numeric cast → set impossible values (< 0 or > 110) to NaN → median impute |
| `bmi` | Zero → NaN (missing marker) → Winsorise at 99th pct / 70.0 → median impute |
| `blood_pressure` | Split "sys/dia" → two columns `systolic`, `diastolic` → median impute |
| `gender` | Map all variants (M/Male/male/1/F/Female/female/0) → binary 0/1 → mode impute |
| `glucose` | Map labels (`"high"` → 200, `"low"` → 55, `"normal"` → 100) → median impute |
| `creatinine` | Abs value (sign flip fix) → cap at 15 mg/dL → median impute |
| `hba1c`, `num_prev_admissions`, etc. | Median impute |
| `diabetes`, `hypertension` | Mode impute (binary) |
| `smoker`, `discharge_disposition` | Mode impute → one-hot encode |
| Duplicates | `drop_duplicates()` keeping first occurrence |

**Rationale:** Median imputation is preferred over mean for skewed clinical variables. Mode imputation is used for binary flags. String labels in glucose are mapped to clinically grounded midpoints rather than dropped, preserving sample size.

---

### 🟡 Sub-step 3 — 3-Layer NumPy Neural Network

**Architecture:**
```
Input (n_features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

**Design decisions:**
- **He initialisation** (`σ = √(2/fan_in)`) — prevents vanishing gradients with ReLU activations
- **ReLU hidden layers** — computationally cheap, gradient-stable for shallow networks
- **Sigmoid output** — produces calibrated probabilities for binary classification
- **Weighted binary cross-entropy** — `pos_weight = neg_rate / pos_rate ≈ 16` to compensate for class imbalance
- **Layer sizes 64→32** — moderate depth avoids overfitting on ~1,600 training samples; larger layers showed no improvement in pilot runs

---

### 🟡 Sub-step 4 — Training & Evaluation

**Training:**
- Full-batch gradient descent, 2000 epochs, learning rate 0.005
- Loss curve plotted to confirm learning (not flatline, not explosive)

**Evaluation metric choice — PR-AUC (Average Precision):**
> With a ~6% readmission rate, accuracy is structurally misleading — a model that predicts "not readmitted" for every patient scores ~94% accuracy while catching zero readmissions. PR-AUC focuses exclusively on the positive class and is robust to extreme imbalance.

**Baseline comparison:**
- `GradientBoostingClassifier` (sklearn) trained on identical split with same scaler
- Both models evaluated with PR-AUC, ROC-AUC, and PR curve

---

### 🟡 Sub-step 5 — Cost-Sensitive Threshold

**Cost assumptions (stated explicitly):**
| Event | Cost |
|---|---|
| False Negative (missed readmission) | ₹15,000 |
| False Positive (unnecessary alert) | ₹1,500 |

The 10:1 ratio reflects the documented evidence that missed readmissions lead to emergency re-hospitalisations far more expensive than proactive discharge planning interventions.

**Method:** Grid search over 500 threshold values (0.01–0.99); select threshold minimising `FN × C_FN + FP × C_FP`.

**Output to Dr. Anand:** Plain-language recommendation with the optimal threshold value, number of patients caught vs. missed, and estimated cost saving vs. the default 0.5 threshold.

---

### 🔴 Sub-step 6 (Hard) — The 94% Accuracy Trap

Demonstrates that a trivial **always-predict-zero** model achieves ~94% accuracy on this dataset:
- Confusion matrix shows zero true positives
- PR-AUC ≈ 0.06 (equal to random baseline)
- Fix: report PR-AUC as the primary metric; show before/after confusion matrices

**Key insight:** On any dataset where the negative class constitutes ≥ 90% of samples, a production-readiness claim based on accuracy alone is not just unhelpful — it is clinically dangerous.

---

### 🔴 Sub-step 7 (Hard) — NN as Feature Extractor

- Extracts 32-dimensional activations from the **penultimate layer** of the trained network
- Trains `LogisticRegression` on these embeddings instead of raw features
- Compares PR-AUC of direct NN classification vs. embedding + logistic pipeline
- **t-SNE visualisation** plots both raw features and learned embeddings coloured by readmission label — visual inspection of cluster separation quality

**What the embeddings learn:** The penultimate ReLU layer captures non-linear combinations of clinical features that are maximally predictive under the training objective. If these embeddings produce better-separated clusters than raw features, it confirms the network has learned a meaningful intermediate representation beyond simple feature scaling.

---

## Scoring Checklist

| Dimension | Status |
|---|---|
| ✅ Sub-steps 1–2 (Easy) | Complete |
| ✅ Sub-steps 3–5 (Medium) | Complete |
| ✅ Sub-steps 6–7 (Hard) | Attempted (Band 4 target) |
| ✅ No hardcoded paths | `DATA_PATH = Path('hospital_records.csv')` |
| ✅ Modular code | ≥ 2 functions per sub-step |
| ✅ No magic numbers | Constants named (`COST_FALSE_NEGATIVE`, `VALID_AGE_MAX`, etc.) |
| ✅ Defensive handling | `try/except` on I/O; shape checks; `errors='coerce'` on all casts |
| ✅ Readable naming | `clean_blood_pressure()`, `compute_expected_cost()`, not `x` or `temp2` |
| ✅ AI Usage Log | Appended as last notebook cell |

---

## Common Pitfalls Avoided

- **Hardcoded paths** — all paths use `pathlib.Path` relative to the working directory
- **Imbalance blindness** — PR-AUC used throughout, not accuracy
- **No stratify on split** — `stratify=y` passed to `train_test_split`
- **Sigmoid overflow** — numerically stable sigmoid implemented with `np.where`
- **ReLU dying neurons** — He initialisation used (not random normal with default std)
- **Gradient vanishing check** — loss curve plotted to verify network learns

---

*Submitted to: `week-08/tuesday/` · Deadline: Wednesday 09:15 AM*
