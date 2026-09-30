import os
import time
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# NUTRIRATE AI - FINAL MODEL + FEATURE IMPORTANCE
# ============================================================

BASE_DIR = r"D:\NutriRate-AML"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")

for folder in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
    os.makedirs(folder, exist_ok=True)


print("=" * 65)
print("       NUTRIRATE AI - FINAL MODEL")
print("=" * 65)


# ============================================================
# 1. LOAD TRAINING AND VALIDATION DATA
# ============================================================

print("\n[1/6] Loading training and validation datasets...")

X_train = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_X_train.parquet")
)

X_val = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_X_val.parquet")
)

y_train = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_y_train.parquet")
)["grade"]

y_val = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_y_val.parquet")
)["grade"]

print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")
print(f"Features           : {X_train.shape[1]}")


# ============================================================
# 2. RECREATE THE SELECTED RANDOM FOREST
# ============================================================

print("\n[2/6] Preparing selected Random Forest...")

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ============================================================
# 3. TRAIN
# ============================================================

print("\n[3/6] Training Random Forest...")
print("This may take approximately 3–5 minutes.")

start = time.time()

model.fit(X_train, y_train)

elapsed = time.time() - start

print(f"Training completed in {elapsed:.2f} seconds.")


# ============================================================
# 4. VALIDATE RECREATED MODEL
# ============================================================

print("\n[4/6] Checking validation performance...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro"
)

print(f"Validation accuracy : {accuracy:.6f}")
print(f"Validation Macro F1 : {macro_f1:.6f}")

print("\nPrevious Random Forest results:")
print("Validation accuracy : 0.887384")
print("Validation Macro F1 : 0.874109")

if (
    abs(accuracy - 0.887384) > 0.001
    or abs(macro_f1 - 0.874109) > 0.001
):
    raise RuntimeError(
        "Recreated model differs from the earlier experiment. "
        "Check the dataset and model parameters before saving."
    )

print("Validation check passed.")


# ============================================================
# 5. SAVE MODEL AND METADATA
# ============================================================

print("\n[5/6] Saving trained model...")

model_path = os.path.join(
    MODEL_DIR,
    "nutrirate_aml_random_forest.joblib"
)

joblib.dump(
    model,
    model_path,
    compress=3
)

metadata = {
    "model": "RandomForestClassifier",
    "features": X_train.columns.tolist(),
    "classes": model.classes_.tolist(),
    "training_samples": len(X_train),
    "validation_accuracy": accuracy,
    "validation_macro_f1": macro_f1,
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42
    }
}

metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Saved:", model_path)
print("Saved:", metadata_path)


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

print("\n[6/6] Generating feature importance...")

forest = model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": forest.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(
    importance_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

importance_df.to_csv(
    os.path.join(TABLE_DIR, "feature_importance.csv"),
    index=False
)

# Plot in ascending order for a readable horizontal chart.
plot_df = importance_df.sort_values("importance")

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["feature"],
    plot_df["importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Mean Decrease in Impurity")
plt.ylabel("Feature")

plt.tight_layout()

figure_path = os.path.join(
    FIGURE_DIR,
    "feature_importance.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 65)
print("              FINALIZATION COMPLETE")
print("=" * 65)

print("\nCreated:")
print("  models/nutrirate_aml_random_forest.joblib")
print("  models/model_metadata.json")
print("  results/tables/feature_importance.csv")
print("  results/figures/feature_importance.png")

print("\nNext: Build the working prediction application.")