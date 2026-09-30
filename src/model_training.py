import os
import time
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# NUTRIRATE AI - ADVANCED ML
# MODEL TRAINING AND COMPARISON
# ============================================================

BASE_DIR = r"D:\NutriRate-AML"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


print("=" * 70)
print("             NUTRIRATE AI - ADVANCED ML")
print("             MODEL TRAINING & COMPARISON")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading AML datasets...")

X_train = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_X_train.parquet")
)

X_val = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_X_val.parquet")
)

X_test = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_X_test.parquet")
)

y_train = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_y_train.parquet")
)["grade"]

y_val = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_y_val.parquet")
)["grade"]

y_test = pd.read_parquet(
    os.path.join(DATA_DIR, "aml_y_test.parquet")
)["grade"]


print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")
print(f"Test samples       : {len(X_test):,}")
print(f"Features           : {X_train.shape[1]}")


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

print("\n[2/8] Preparing models...")


# ------------------------------------------------------------
# BASELINE
# Logistic Regression
# ------------------------------------------------------------

logistic_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# BASELINE / ENSEMBLE
# Random Forest
# ------------------------------------------------------------

random_forest = Pipeline([
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


# ------------------------------------------------------------
# ADVANCED MODEL
# Histogram Gradient Boosting
# ------------------------------------------------------------

hist_gradient_boosting = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    )
])


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest,
    "HistGradientBoosting": hist_gradient_boosting
}


# ============================================================
# 3. TRAIN MODELS
# ============================================================

print("\n[3/8] Training models...")
print("-" * 70)

trained_models = {}
validation_results = []


for name, model in models.items():

    print(f"\nTraining: {name}")

    start = time.time()

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    trained_models[name] = model

    val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_pred)

    macro_precision = precision_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )

    validation_results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": elapsed
    })

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Training time  : {elapsed:.2f} seconds")


# ============================================================
# 4. VALIDATION COMPARISON
# ============================================================

print("\n[4/8] Comparing validation performance...")

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    "macro_f1",
    ascending=False
)

print("\nValidation Results:")
print(
    validation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

validation_df.to_csv(
    os.path.join(TABLE_DIR, "validation_model_comparison.csv"),
    index=False
)


# ============================================================
# 5. SELECT MODEL
# ============================================================

best_model_name = validation_df.iloc[0]["model"]

best_model = trained_models[best_model_name]

print("\nSelected model based on validation Macro F1:")
print(best_model_name)


# ============================================================
# 6. FINAL TEST EVALUATION
# ============================================================

print("\n[5/8] Evaluating models on TEST set...")
print("-" * 70)

test_results = []

for name, model in trained_models.items():

    test_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    macro_precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    test_results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    print(f"\n{name}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")


test_df = pd.DataFrame(test_results)

test_df = test_df.sort_values(
    "macro_f1",
    ascending=False
)

test_df.to_csv(
    os.path.join(TABLE_DIR, "test_model_comparison.csv"),
    index=False
)


# ============================================================
# 7. DETAILED ERROR ANALYSIS
# ============================================================

print("\n[6/8] Performing error analysis...")

best_test_pred = best_model.predict(X_test)

labels = ["A", "B", "C", "D", "E"]


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report = classification_report(
    y_test,
    best_test_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(TABLE_DIR, "classification_report.csv")
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_test_pred,
        labels=labels,
        zero_division=0
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    best_test_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Predicted_{x}" for x in labels]
)

cm_df.to_csv(
    os.path.join(TABLE_DIR, "confusion_matrix.csv")
)

print("\nConfusion Matrix:")
print(cm_df)


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

predictions = pd.DataFrame({
    "actual": y_test.values,
    "predicted": best_test_pred
})

predictions["correct"] = (
    predictions["actual"] ==
    predictions["predicted"]
)

predictions.to_csv(
    os.path.join(PRED_DIR, "best_model_predictions.csv"),
    index=False
)


# ------------------------------------------------------------
# Misclassification analysis
# ------------------------------------------------------------

errors = predictions[
    predictions["correct"] == False
].copy()

print("\nError Analysis:")
print(f"Total test samples : {len(predictions):,}")
print(f"Correct predictions: {predictions['correct'].sum():,}")
print(f"Incorrect          : {len(errors):,}")

if len(errors) > 0:

    error_pairs = (
        errors
        .groupby(["actual", "predicted"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\nMost common misclassifications:")
    print(error_pairs.head(15).to_string(index=False))

    error_pairs.to_csv(
        os.path.join(TABLE_DIR, "misclassification_pairs.csv"),
        index=False
    )


# ============================================================
# 8. SAVE FINAL SUMMARY
# ============================================================

print("\n[7/8] Saving experiment summary...")

summary = {
    "dataset": "NutriRate AI Clean Dataset",
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "feature_count": int(X_train.shape[1]),
    "classes": labels,
    "models": list(models.keys()),
    "selected_model_validation": best_model_name,
    "selection_metric": "Macro F1"
}

with open(
    os.path.join(TABLE_DIR, "experiment_summary.json"),
    "w"
) as f:
    json.dump(summary, f, indent=4)


print("\n[8/8] Complete.")

print("\n" + "=" * 70)
print("                 MODELING COMPLETE")
print("=" * 70)

print("\nValidation comparison:")
print(validation_df.to_string(index=False))

print("\nTest comparison:")
print(test_df.to_string(index=False))

print("\nSelected model:")
print(best_model_name)

print("\nResults saved in:")
print("  results/tables/")
print("  results/predictions/")

print("\nNext step:")
print("  Feature importance + visualizations + final report")

print("=" * 70)