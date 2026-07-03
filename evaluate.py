import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from data_loader import load_test_data

# ==========================================================
# Command Line Arguments
# ==========================================================

parser = argparse.ArgumentParser(
    description="Evaluate Federated Learning Model"
)

parser.add_argument(
    "--dataset",
    choices=["iid", "noniid"],
    required=True,
    help="Dataset type",
)

parser.add_argument(
    "--experiment",
    required=True,
    help="Experiment name (e.g. fedavg_iid)",
)

args = parser.parse_args()

DATASET = args.dataset
EXPERIMENT = args.experiment

# ==========================================================
# Experiment Directories
# ==========================================================

BASE_DIR = os.path.join(
    "experiments",
    EXPERIMENT,
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "result_plots",
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading trained global model...")
print("=" * 60)

model = tf.keras.models.load_model(

    os.path.join(

        MODELS_DIR,

        "global_model.keras",

    )

)

# ==========================================================
# Load Test Dataset
# ==========================================================

X_test, y_test = load_test_data(DATASET)

print(f"\nDataset     : {DATASET}")
print(f"Experiment  : {EXPERIMENT}")
print(f"Test Samples: {len(X_test)}")

# ==========================================================
# Predict
# ==========================================================

print("\nGenerating predictions...")

y_prob = model.predict(
    X_test,
    verbose=0,
)

y_prob = y_prob.flatten()

y_pred = (y_prob >= 0.5).astype(int)

# ==========================================================
# Calculate Metrics
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
)

recall = recall_score(
    y_test,
    y_pred,
)

f1 = f1_score(
    y_test,
    y_pred,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)

# ==========================================================
# Print Metrics
# ==========================================================

print("\n")
print("=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
    )
)

# ==========================================================
# Save Metrics
# ==========================================================

metrics = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-score",

        "ROC-AUC",

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        roc_auc,

    ],

})

metrics.to_csv(

    os.path.join(

        RESULTS_DIR,

        "federated_metrics.csv",

    ),

    index=False,

)

# ==========================================================
# Save Predictions
# ==========================================================

predictions = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred,

    "Probability": y_prob,

})

predictions.to_csv(

    os.path.join(

        RESULTS_DIR,

        "predictions.csv",

    ),

    index=False,

)

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
)

disp.plot(
    cmap="Blues",
    colorbar=False,
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confusion_matrix.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

print("Generating ROC curve...")

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob,
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}",
)

plt.plot(
    [0,1],
    [0,1],
    "--",
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "roc_curve.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Precision-Recall Curve
# ==========================================================

print("Generating Precision-Recall curve...")

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob,
)

plt.figure(figsize=(6,6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2,
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "precision_recall_curve.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Metrics Bar Chart
# ==========================================================

print("Generating metrics chart...")

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "ROC-AUC",

]

metric_values = [

    accuracy,

    precision,

    recall,

    f1,

    roc_auc,

]

plt.figure(figsize=(8,5))

plt.bar(
    metric_names,
    metric_values,
)

plt.ylim(0,1)

plt.ylabel("Score")

plt.title("Performance Metrics")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "metrics_bar_chart.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Prediction Probability Distribution
# ==========================================================

print("Generating prediction probability histogram...")

plt.figure(figsize=(8,5))

plt.hist(
    y_prob,
    bins=30,
)

plt.xlabel("Predicted Probability")

plt.ylabel("Frequency")

plt.title("Prediction Probability Distribution")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "probability_distribution.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Finished
# ==========================================================

print("\n")
print("=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print(f"\nExperiment : {EXPERIMENT}")
print(f"Dataset    : {DATASET}")

print("\nSaved files")

print("-" * 60)

print(os.path.join(
    RESULTS_DIR,
    "federated_metrics.csv",
))

print(os.path.join(
    RESULTS_DIR,
    "predictions.csv",
))

print(os.path.join(
    PLOTS_DIR,
    "confusion_matrix.png",
))

print(os.path.join(
    PLOTS_DIR,
    "roc_curve.png",
))

print(os.path.join(
    PLOTS_DIR,
    "precision_recall_curve.png",
))

print(os.path.join(
    PLOTS_DIR,
    "metrics_bar_chart.png",
))

print(os.path.join(
    PLOTS_DIR,
    "probability_distribution.png",
))

print("\nDone.")
