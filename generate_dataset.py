"""
generate_dataset.py
-------------------
Generates a synthetic hospital_records.csv (2,000 rows) that mimics the
messy real-world dataset described in the W8-Tuesday assignment.

Intentional data quality issues introduced:
  - Age column: some negative values, some > 120, some stored as strings ("45 yrs")
  - BMI column: some 0.0, some outliers (> 80), some NaN
  - Blood pressure: mixed formats ("120/80" vs separate columns)
  - Gender: inconsistent categories ("M", "Male", "male", "F", "Female", "female", 1, 0)
  - Missing values scattered across multiple columns
  - Duplicate rows (~2%)
  - Readmission_30d: target variable (imbalanced ~8% positive)
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
N = 2000

# ── Helper ──────────────────────────────────────────────────────────────────

def introduce_missing(series, frac, rng):
    mask = rng.random(len(series)) < frac
    series = series.astype(object)
    series[mask] = np.nan
    return series


# ── Core features (clean) ────────────────────────────────────────────────────

ages_true      = rng.integers(18, 90, size=N).astype(float)
bmi_true       = rng.normal(27, 5, size=N).clip(15, 55)
systolic       = rng.integers(90, 180, size=N)
diastolic      = rng.integers(60, 110, size=N)
glucose        = rng.normal(105, 25, size=N).clip(60, 300)
hba1c          = rng.normal(6.2, 1.2, size=N).clip(4.5, 14)
creatinine     = rng.normal(1.1, 0.4, size=N).clip(0.4, 8)
num_prev_admissions = rng.integers(0, 10, size=N)
length_of_stay = rng.integers(1, 30, size=N)
num_medications= rng.integers(1, 20, size=N)

gender_clean   = rng.choice(["Male", "Female"], size=N, p=[0.52, 0.48])
diabetes       = (rng.random(N) < 0.18).astype(int)
hypertension   = (rng.random(N) < 0.30).astype(int)
smoker         = rng.choice(["Yes", "No", "Former"], size=N, p=[0.22, 0.60, 0.18])
discharge_disp = rng.choice(["Home", "SNF", "Rehab", "AMA"], size=N,
                             p=[0.70, 0.15, 0.10, 0.05])

# ── Target: ~8% readmission (imbalanced) ────────────────────────────────────
log_odds = (
    -4.0
    + 0.02  * (ages_true - 60)
    + 0.05  * (bmi_true - 27)
    + 0.40  * num_prev_admissions
    + 0.30  * diabetes
    + 0.25  * hypertension
    - 0.10  * length_of_stay
    + 0.20  * (discharge_disp == "AMA").astype(float)
)
prob = 1 / (1 + np.exp(-log_odds))
readmission = (rng.random(N) < prob).astype(int)

# ── Introduce dirty data ─────────────────────────────────────────────────────

# Age: negatives, > 120, string noise
ages_dirty = ages_true.copy()
neg_idx = rng.choice(N, 30, replace=False)
ages_dirty[neg_idx] = -rng.integers(1, 30, size=30)          # negative ages
old_idx = rng.choice(N, 20, replace=False)
ages_dirty[old_idx] = rng.integers(121, 150, size=20)         # impossible ages
ages_series = pd.Series(ages_dirty.astype(object))
str_idx = rng.choice(N, 40, replace=False)
ages_series.iloc[str_idx] = [f"{int(a)} yrs" for a in ages_true[str_idx]]
ages_series = introduce_missing(ages_series, 0.03, rng)

# BMI: zeros, extreme outliers, missing
bmi_dirty = bmi_true.copy()
zero_idx = rng.choice(N, 35, replace=False)
bmi_dirty[zero_idx] = 0.0
outlier_idx = rng.choice(N, 15, replace=False)
bmi_dirty[outlier_idx] = rng.uniform(85, 120, size=15)
bmi_series = introduce_missing(pd.Series(bmi_dirty), 0.04, rng)

# Blood pressure: combined string column
bp_combined = pd.Series([f"{s}/{d}" for s, d in zip(systolic, diastolic)])
bp_combined = introduce_missing(bp_combined, 0.03, rng)

# Gender: inconsistent categories
gender_map = {
    "Male":   ["Male", "male", "M", "m", "1"],
    "Female": ["Female", "female", "F", "f", "0"],
}
gender_dirty = []
for g in gender_clean:
    choices = gender_map[g]
    gender_dirty.append(rng.choice(choices))
gender_series = introduce_missing(pd.Series(gender_dirty), 0.02, rng)

# Glucose: a few string entries ("high", "low")
glucose_series = pd.Series(glucose.copy().astype(object))
str_gluc = rng.choice(N, 15, replace=False)
glucose_series.iloc[str_gluc] = rng.choice(["high", "low", "normal"], size=15)
glucose_series = introduce_missing(glucose_series, 0.05, rng)

# Creatinine: a few negatives
creat_series = pd.Series(creatinine.copy())
neg_creat = rng.choice(N, 10, replace=False)
creat_series.iloc[neg_creat] = -creat_series.iloc[neg_creat]
creat_series = introduce_missing(creat_series, 0.03, rng)

# HbA1c: missing
hba1c_series = introduce_missing(pd.Series(hba1c), 0.06, rng)

# Other columns
num_prev_series = introduce_missing(pd.Series(num_prev_admissions.astype(float)), 0.02, rng)
los_series      = introduce_missing(pd.Series(length_of_stay.astype(float)), 0.01, rng)
meds_series     = introduce_missing(pd.Series(num_medications.astype(float)), 0.02, rng)

# ── Assemble DataFrame ───────────────────────────────────────────────────────

df = pd.DataFrame({
    "patient_id":            [f"P{str(i).zfill(5)}" for i in range(1, N + 1)],
    "age":                   ages_series,
    "gender":                gender_series,
    "bmi":                   bmi_series,
    "blood_pressure":        bp_combined,
    "glucose":               glucose_series,
    "hba1c":                 hba1c_series,
    "creatinine":            creat_series,
    "diabetes":              introduce_missing(pd.Series(diabetes.astype(float)), 0.02, rng),
    "hypertension":          introduce_missing(pd.Series(hypertension.astype(float)), 0.02, rng),
    "smoker":                introduce_missing(pd.Series(smoker), 0.02, rng),
    "num_prev_admissions":   num_prev_series,
    "length_of_stay":        los_series,
    "num_medications":       meds_series,
    "discharge_disposition": introduce_missing(pd.Series(discharge_disp), 0.02, rng),
    "readmission_30d":       readmission,
})

# ── Inject ~2% duplicate rows ────────────────────────────────────────────────
dup_idx = rng.choice(N, int(N * 0.02), replace=False)
dups    = df.iloc[dup_idx].copy()
df      = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=SEED)
df      = df.reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv("hospital_records.csv", index=False)
print(f"Saved hospital_records.csv  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Readmission rate: {df['readmission_30d'].mean():.2%}")
print("Missing value counts:\n", df.isnull().sum())
