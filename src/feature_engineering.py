import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# NUTRIRATE AI - ADVANCED ML
# Feature Engineering + Train/Validation/Test Split
# ============================================================

BASE_DIR = r"D:\NutriRate-AML"

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "nutrirate_ml_clean.parquet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

TABLE_DIR = os.path.join(
    BASE_DIR, "results", "tables"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


print("=" * 65)
print("          NUTRIRATE AI - ADVANCED ML")
print("          FEATURE ENGINEERING PIPELINE")
print("=" * 65)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/9] Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

print("\n[2/9] Defining features and target...")

base_features = [
    "energy",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "protein",
    "fiber",
    "salt"
]

target = "grade"

# IMPORTANT:
# nutriscore_score is NOT used as an input feature.
# It is directly related to the target grade and can cause
# target leakage / near-leakage.

print("\nBase features:")
for feature in base_features:
    print("  -", feature)

print("\nTarget:", target)
print("Excluded from predictors: nutriscore_score")


# ============================================================
# 3. CREATE ADVANCED FEATURES
# ============================================================

print("\n[3/9] Creating nutritional ratio features...")

X = df[base_features].copy()
y = df[target].copy()

EPS = 1e-6


def safe_ratio(numerator, denominator):
    """
    Calculate a stable ratio.
    If denominator is zero or extremely close to zero,
    return NaN instead of producing extreme values.
    """
    denominator = denominator.astype(float)

    result = np.where(
        np.abs(denominator) > EPS,
        numerator / denominator,
        np.nan
    )

    return result


# ------------------------------------------------------------
# Nutritional composition ratios
# ------------------------------------------------------------

X["saturated_fat_ratio"] = safe_ratio(
    X["saturated_fat"],
    X["fat"]
)

X["sugar_carb_ratio"] = safe_ratio(
    X["sugars"],
    X["carbohydrates"]
)

X["fiber_carb_ratio"] = safe_ratio(
    X["fiber"],
    X["carbohydrates"]
)

X["protein_carb_ratio"] = safe_ratio(
    X["protein"],
    X["carbohydrates"]
)

# ------------------------------------------------------------
# Nutrient-to-energy ratios
# ------------------------------------------------------------

X["protein_energy_ratio"] = safe_ratio(
    X["protein"],
    X["energy"]
)

X["fat_energy_ratio"] = safe_ratio(
    X["fat"],
    X["energy"]
)

X["sugar_energy_ratio"] = safe_ratio(
    X["sugars"],
    X["energy"]
)

X["fiber_energy_ratio"] = safe_ratio(
    X["fiber"],
    X["energy"]
)

X["salt_energy_ratio"] = safe_ratio(
    X["salt"],
    X["energy"]
)


print(f"Total features after engineering: {X.shape[1]}")

print("\nEngineered features:")
for feature in X.columns:
    if feature not in base_features:
        print("  -", feature)


# ============================================================
# 4. CLEAN INF / NaN VALUES
# ============================================================

print("\n[4/9] Checking feature quality...")

X = X.replace([np.inf, -np.inf], np.nan)

missing_before = X.isna().sum()

print("\nMissing values:")
print(missing_before[missing_before > 0])

# We intentionally do NOT fill missing values here.
# Missing-value handling will be performed inside the
# training pipeline using only training data.


# ============================================================
# 5. REMOVE INVALID TARGET ROWS
# ============================================================

print("\n[5/9] Validating target...")

valid_grades = ["A", "B", "C", "D", "E"]

valid_mask = y.isin(valid_grades)

X = X.loc[valid_mask].reset_index(drop=True)
y = y.loc[valid_mask].reset_index(drop=True)

print(f"Valid samples: {len(y):,}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nTarget distribution (%):")
print(
    (y.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[6/9] Creating stratified train/validation/test split...")

# First: 70% train + 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second: split temporary into 15% validation + 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)


print("\nSplit sizes:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")

print("\nSplit percentages:")
total = len(X_train) + len(X_val) + len(X_test)

print(f"Train      : {len(X_train) / total * 100:.2f}%")
print(f"Validation : {len(X_val) / total * 100:.2f}%")
print(f"Test       : {len(X_test) / total * 100:.2f}%")


# ============================================================
# 7. VERIFY STRATIFICATION
# ============================================================

print("\n[7/9] Verifying class distribution...")

distribution = pd.DataFrame({
    "Train": y_train.value_counts(normalize=True),
    "Validation": y_val.value_counts(normalize=True),
    "Test": y_test.value_counts(normalize=True)
}) * 100

distribution = distribution.round(2)

print("\nClass distribution (%):")
print(distribution)


# ============================================================
# 8. SAVE DATASETS
# ============================================================

print("\n[8/9] Saving AML datasets...")

X_train.to_parquet(
    os.path.join(OUTPUT_DIR, "aml_X_train.parquet"),
    index=False
)

X_val.to_parquet(
    os.path.join(OUTPUT_DIR, "aml_X_val.parquet"),
    index=False
)

X_test.to_parquet(
    os.path.join(OUTPUT_DIR, "aml_X_test.parquet"),
    index=False
)

y_train.to_frame(name=target).to_parquet(
    os.path.join(OUTPUT_DIR, "aml_y_train.parquet"),
    index=False
)

y_val.to_frame(name=target).to_parquet(
    os.path.join(OUTPUT_DIR, "aml_y_val.parquet"),
    index=False
)

y_test.to_frame(name=target).to_parquet(
    os.path.join(OUTPUT_DIR, "aml_y_test.parquet"),
    index=False
)


# ============================================================
# 9. SAVE FEATURE DEFINITIONS
# ============================================================

print("\n[9/9] Saving feature definition...")

feature_rows = []

for feature in base_features:
    feature_rows.append({
        "feature": feature,
        "type": "original",
        "description": "Original nutritional attribute"
    })

feature_descriptions = {
    "saturated_fat_ratio":
        "Saturated fat relative to total fat",

    "sugar_carb_ratio":
        "Sugar relative to total carbohydrates",

    "fiber_carb_ratio":
        "Fiber relative to total carbohydrates",

    "protein_carb_ratio":
        "Protein relative to total carbohydrates",

    "protein_energy_ratio":
        "Protein relative to energy",

    "fat_energy_ratio":
        "Fat relative to energy",

    "sugar_energy_ratio":
        "Sugar relative to energy",

    "fiber_energy_ratio":
        "Fiber relative to energy",

    "salt_energy_ratio":
        "Salt relative to energy"
}

for feature, description in feature_descriptions.items():
    feature_rows.append({
        "feature": feature,
        "type": "engineered",
        "description": description
    })

feature_definition = pd.DataFrame(feature_rows)

feature_definition.to_csv(
    os.path.join(TABLE_DIR, "aml_feature_definition.csv"),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("                 PIPELINE COMPLETE")
print("=" * 65)

print("\nFeatures:", X.shape[1])

print(f"Train samples      : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")
print(f"Test samples       : {len(X_test):,}")

print("\nFiles created:")

files = [
    "aml_X_train.parquet",
    "aml_X_val.parquet",
    "aml_X_test.parquet",
    "aml_y_train.parquet",
    "aml_y_val.parquet",
    "aml_y_test.parquet"
]

for file in files:
    print("  data/processed/" + file)

print("  results/tables/aml_feature_definition.csv")

print("\nNext step: model training and comparison.")
print("=" * 65)