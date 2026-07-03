import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Command Line Arguments
# ==========================================================

parser = argparse.ArgumentParser(
    description="Plot Federated Learning Training Curves"
)

parser.add_argument(
    "--experiment",
    required=True,
    help="Experiment name (e.g. fedavg_iid)",
)

args = parser.parse_args()

EXPERIMENT = args.experiment

# ==========================================================
# Directories
# ==========================================================

BASE_DIR = os.path.join(
    "experiments",
    EXPERIMENT,
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "result_plots",
)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================================
# Load Training History
# ==========================================================

history_file = os.path.join(
    RESULTS_DIR,
    "training_history.csv",
)

if not os.path.exists(history_file):
    raise FileNotFoundError(
        f"Training history not found:\n{history_file}"
    )

history = pd.read_csv(history_file)

print("=" * 60)
print("Loaded Training History")
print("=" * 60)

print(history.head())

# ==========================================================
# Training Loss
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history["Round"],
    history["Loss"],
    marker="o",
    linewidth=2,
)

plt.xlabel("Communication Round")

plt.ylabel("Training Loss")

plt.title(
    f"Training Loss ({EXPERIMENT})"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "training_loss.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Training Accuracy
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history["Round"],
    history["Accuracy"],
    marker="o",
    linewidth=2,
)

plt.xlabel("Communication Round")

plt.ylabel("Training Accuracy")

plt.title(
    f"Training Accuracy ({EXPERIMENT})"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "training_accuracy.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Combined Training Curves
# ==========================================================

fig, ax1 = plt.subplots(figsize=(9,5))

ax1.plot(
    history["Round"],
    history["Loss"],
    marker="o",
    linewidth=2,
    label="Loss",
)

ax1.set_xlabel("Communication Round")

ax1.set_ylabel("Loss")

ax2 = ax1.twinx()

ax2.plot(
    history["Round"],
    history["Accuracy"],
    marker="s",
    linewidth=2,
)

ax2.set_ylabel("Accuracy")

plt.title(
    f"Federated Learning Training Curves ({EXPERIMENT})"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "training_curves.png",
    ),
    dpi=600,
)

plt.close()

# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)
print("Training plots generated successfully.")
print("=" * 60)

print("\nSaved files:")

print(
    os.path.join(
        PLOTS_DIR,
        "training_loss.png",
    )
)

print(
    os.path.join(
        PLOTS_DIR,
        "training_accuracy.png",
    )
)

print(
    os.path.join(
        PLOTS_DIR,
        "training_curves.png",
    )
)
