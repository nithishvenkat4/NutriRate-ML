import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ============================================================
# NUTRIRATE AI - RESULT VISUALIZATION
# ============================================================

BASE_DIR = r"D:\NutriRate-AML"

TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")
PRED_DIR = os.path.join(BASE_DIR, "results", "predictions")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


print("=" * 65)
print("       NUTRIRATE AI - RESULT VISUALIZATION")
print("=" * 65)


# ============================================================
# 1. MODEL COMPARISON
# ============================================================

print("\n[1/5] Creating model comparison chart...")

results = pd.read_csv(
    os.path.join(TABLE_DIR, "test_model_comparison.csv")
)

metrics = [
    "accuracy",
    "macro_f1",
    "weighted_f1"
]

plot_df = results.melt(
    id_vars="model",
    value_vars=metrics,
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_df,
    x="model",
    y="score",
    hue="metric"
)

plt.ylim(0, 1)
plt.title("NutriRate AI - Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "model_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

print("[2/5] Creating confusion matrix...")

predictions = pd.read_csv(
    os.path.join(PRED_DIR, "best_model_predictions.csv")
)

labels = ["A", "B", "C", "D", "E"]

cm = confusion_matrix(
    predictions["actual"],
    predictions["predicted"],
    labels=labels
)

plt.figure(figsize=(8, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt=",",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. NORMALIZED CONFUSION MATRIX
# ============================================================

print("[3/5] Creating normalized confusion matrix...")

cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 7))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Random Forest Normalized Confusion Matrix")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "confusion_matrix_normalized.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("[4/5] Creating class distribution chart...")

train_y = pd.read_parquet(
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "aml_y_train.parquet"
    )
)["grade"]

class_counts = train_y.value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title("Training Dataset - Nutri-Score Grade Distribution")
plt.xlabel("Nutri-Score Grade")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. ERROR DISTRIBUTION
# ============================================================

print("[5/5] Creating error distribution chart...")

errors = predictions[
    predictions["actual"] != predictions["predicted"]
].copy()

error_pairs = (
    errors
    .groupby(["actual", "predicted"])
    .size()
    .reset_index(name="count")
)

error_pairs["pair"] = (
    error_pairs["actual"]
    + " → "
    + error_pairs["predicted"]
)

error_pairs = error_pairs.sort_values(
    "count",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    error_pairs["pair"],
    error_pairs["count"]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Random Forest Misclassification Patterns")
plt.xlabel("Number of Errors")
plt.ylabel("Actual → Predicted")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "top_misclassification_patterns.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("             FIGURES GENERATED")
print("=" * 65)

print("\nSaved to:")
print(FIGURE_DIR)

print("\nFiles:")

for filename in [
    "model_comparison.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "class_distribution.png",
    "top_misclassification_patterns.png"
]:
    print("  -", filename)

print("\nDone.")