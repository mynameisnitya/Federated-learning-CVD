import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Configuration
# ==========================================================

EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "comparison_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Find Experiments Automatically
# ==========================================================

experiment_folders = sorted([

    folder

    for folder in os.listdir(EXPERIMENTS_DIR)

    if os.path.isdir(
        os.path.join(
            EXPERIMENTS_DIR,
            folder,
        )
    )

])

print("=" * 60)
print("Experiments Found")
print("=" * 60)

for folder in experiment_folders:
    print(folder)

# ==========================================================
# Load Metrics
# ==========================================================

results = []

for folder in experiment_folders:

    metrics_file = os.path.join(
        EXPERIMENTS_DIR,
        folder,
        "results",
        "federated_metrics.csv",
    )

    if not os.path.exists(metrics_file):

        print(f"Skipping {folder}")

        continue

    metrics = pd.read_csv(metrics_file)

    metric_dict = {
        row["Metric"]: row["Value"]
        for _, row in metrics.iterrows()
    }

    metric_dict["Experiment"] = folder

    results.append(metric_dict)

comparison = pd.DataFrame(results)

comparison = comparison[
    [
        "Experiment",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]
]

comparison = comparison.sort_values("Experiment")

comparison.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "comparison_metrics.csv",
    ),

    index=False,

)

print("\nComparison Table\n")

print(comparison)

# ==========================================================
# Helper Function
# ==========================================================

def plot_metric(metric, filename):

    plt.figure(figsize=(8,5))

    plt.bar(
        comparison["Experiment"],
        comparison[metric],
    )

    plt.ylim(0,1)

    plt.ylabel(metric)

    plt.title(f"{metric} Comparison")

    plt.xticks(rotation=20, ha="right")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            filename,
        ),

        dpi=600,

    )

    plt.close()

# ==========================================================
# Individual Plots
# ==========================================================

plot_metric(
    "Accuracy",
    "accuracy_comparison.png",
)

plot_metric(
    "Precision",
    "precision_comparison.png",
)

plot_metric(
    "Recall",
    "recall_comparison.png",
)

plot_metric(
    "F1-score",
    "f1_comparison.png",
)

plot_metric(
    "ROC-AUC",
    "roc_auc_comparison.png",
)

# ==========================================================
# Publication Quality Grouped Comparison
# ==========================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1-score",

    "ROC-AUC",

]

x = np.arange(len(comparison))

width = 0.15

plt.figure(figsize=(12,6))

for i, metric in enumerate(metrics):

    plt.bar(

        x + i * width,

        comparison[metric],

        width,

        label=metric,

    )

plt.xticks(

    x + width * 2,

    comparison["Experiment"],

    rotation=20,

    ha="right",

)

plt.ylim(0,1)

plt.ylabel("Score")

plt.title("Performance Comparison Across Federated Learning Experiments")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "grouped_metrics_comparison.png",
    ),

    dpi=600,

)

plt.close()

# ==========================================================
# Heatmap-style Table
# ==========================================================

fig, ax = plt.subplots(figsize=(8,3))

ax.axis("off")

table = ax.table(

    cellText=np.round(
        comparison.iloc[:,1:].values,
        4,
    ),

    rowLabels=comparison["Experiment"],

    colLabels=comparison.columns[1:],

    cellLoc="center",

    loc="center",

)

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1.2,1.5)

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "comparison_table.png",
    ),

    dpi=600,

)

plt.close()

# ==========================================================
# Finished
# ==========================================================

print("\n")
print("=" * 60)
print("Comparison completed successfully.")
print("=" * 60)

print("\nGenerated files:\n")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(file)
