import duckdb
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = r"D:\NutriRateAI\nutrirate_ann_dataset.parquet"
OUTPUT_DIR = r"D:\NutriRateAI"

FEATURES = [
    "energy",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "protein",
    "fiber",
    "salt"
]

GRADE_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("NUTRIRATE AI - ML DATA PREPARATION")
print("=" * 75)

print("\nLoading ANN dataset...")

query = f"""
SELECT
    energy,
    fat,
    saturated_fat,
    carbohydrates,
    sugars,
    protein,
    fiber,
    salt,
    grade
FROM read_parquet('{INPUT}')
"""

df = duckdb.query(query).to_df()

print(f"Loaded rows: {len(df):,}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURES].copy()

y = df["grade"].map(GRADE_MAPPING)

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nTarget mapping:")
for grade, value in GRADE_MAPPING.items():
    print(f"  {grade} -> {value}")


# ============================================================
# FIRST SPLIT
# ============================================================
# 70% training
# 30% temporary
#
# Stratification keeps A/B/C/D/E proportions consistent.

print("\nCreating train/test split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# SECOND SPLIT
# ============================================================
# Temporary 30% is divided equally:
#
# 15% validation
# 15% test

print("Creating validation/test split...")

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nFitting StandardScaler on training data only...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# CONVERT TO DATAFRAMES
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=FEATURES
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=FEATURES
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=FEATURES
)


# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving prepared datasets...")

X_train_scaled.to_parquet(
    f"{OUTPUT_DIR}\\X_train.parquet",
    index=False
)

X_val_scaled.to_parquet(
    f"{OUTPUT_DIR}\\X_val.parquet",
    index=False
)

X_test_scaled.to_parquet(
    f"{OUTPUT_DIR}\\X_test.parquet",
    index=False
)

y_train.to_frame("grade").to_parquet(
    f"{OUTPUT_DIR}\\y_train.parquet",
    index=False
)

y_val.to_frame("grade").to_parquet(
    f"{OUTPUT_DIR}\\y_val.parquet",
    index=False
)

y_test.to_frame("grade").to_parquet(
    f"{OUTPUT_DIR}\\y_test.parquet",
    index=False
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}\\nutrirate_scaler.pkl"
)


# ============================================================
# REPORT DATASET SIZES
# ============================================================

print("\n" + "=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print(f"\nTraining:   {len(X_train):,}")
print(f"Validation: {len(X_val):,}")
print(f"Test:       {len(X_test):,}")
print(f"Total:      {len(X_train) + len(X_val) + len(X_test):,}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")
print("-" * 75)

for name, labels in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:

    print(f"\n{name}")

    counts = labels.value_counts().sort_index()

    for numeric, count in counts.items():

        grade = list(GRADE_MAPPING.keys())[numeric]

        percentage = count / len(labels) * 100

        print(
            f"  {grade}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# VERIFY SCALING
# ============================================================

print("\nTRAINING FEATURE MEANS AFTER SCALING")
print("-" * 75)

for feature in FEATURES:

    mean = X_train_scaled[feature].mean()
    std = X_train_scaled[feature].std()

    print(
        f"{feature:20s} "
        f"mean={mean:.6f} "
        f"std={std:.6f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("ML DATA PREPARATION COMPLETE")
print("=" * 75)

print("\nFiles created:")

files = [
    "X_train.parquet",
    "X_val.parquet",
    "X_test.parquet",
    "y_train.parquet",
    "y_val.parquet",
    "y_test.parquet",
    "nutrirate_scaler.pkl"
]

for file in files:
    print(f"  {OUTPUT_DIR}\\{file}")