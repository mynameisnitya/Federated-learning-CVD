import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

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

from tensorflow.keras.callbacks import (
    EarlyStopping,
)

from model import create_model

# ==========================================================
# Configuration
# ==========================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

LOCAL_EPOCHS = 5

COMMUNICATION_ROUNDS = 20

BATCH_SIZE = 64

# ==========================================================
# Command Line Arguments
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(

    "--experiment",

    choices=[
        "centralised",
        "simulated",
    ],

    required=True,

)

parser.add_argument(

    "--dataset",

    choices=[
        "iid",
        "noniid",
    ],

    required=True,

)

args = parser.parse_args()

EXPERIMENT = args.experiment

DATASET = args.dataset

EXPERIMENT_NAME = f"{EXPERIMENT}_{DATASET}"

print("=" * 60)

print(EXPERIMENT_NAME.upper())

print("=" * 60)

# ==========================================================
# Output Directories
# ==========================================================

BASE_DIR = os.path.join(

    "experiments",

    EXPERIMENT_NAME,

)

MODEL_DIR = os.path.join(

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

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

TRAIN_FILE = os.path.join(

    "datasets",

    f"{DATASET}_train_global.csv",

)

TEST_FILE = os.path.join(

    "datasets",

    f"{DATASET}_test_global.csv",

)

train_df = pd.read_csv(TRAIN_FILE)

test_df = pd.read_csv(TEST_FILE)

FEATURES = [

    "AGE",

    "GENDER",

    "GLUCOSE",

    "CHOLESTEROL",

    "HEART_RATE",

    "SYS_BP",

    "DIA_BP",

]

TARGET = "CVD"

X_train = train_df[FEATURES].values.astype("float32")

y_train = train_df[TARGET].values.astype("float32")

X_test = test_df[FEATURES].values.astype("float32")

y_test = test_df[TARGET].values.astype("float32")

print(f"Training samples : {len(X_train)}")

print(f"Testing samples  : {len(X_test)}")

# ==========================================================
# Load Hospital Datasets (only needed for simulation)
# ==========================================================

hospital_data = []

if EXPERIMENT == "simulated":

    for i in range(1,4):

        df = pd.read_csv(

            os.path.join(

                "datasets",

                f"{DATASET}_hospital{i}.csv",

            )

        )

        X = df[FEATURES].values.astype("float32")

        y = df[TARGET].values.astype("float32")

        hospital_data.append((X,y))

    print()

    print("Loaded 3 simulated hospitals.")

# ==========================================================
# Class Weights
# ==========================================================

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train,

)

CLASS_WEIGHTS = dict(

    zip(

        np.unique(y_train),

        weights,

    )

)

print()

print("Class Weights")

print(CLASS_WEIGHTS)

# ==========================================================
# Create Model
# ==========================================================

model = create_model()

print()

model.summary()

# ==========================================================
# Utility Functions
# ==========================================================

def save_training_history(history, filename="training_history.csv"):

    history_df = pd.DataFrame({

        "Epoch": range(
            1,
            len(history.history["loss"]) + 1,
        ),

        "Loss": history.history["loss"],

        "Validation Loss": history.history["val_loss"],

        "Accuracy": history.history["accuracy"],

        "Validation Accuracy": history.history["val_accuracy"],

    })

    history_df.to_csv(

        os.path.join(
            RESULTS_DIR,
            filename,
        ),

        index=False,

    )

    return history_df


# ==========================================================

def plot_training_curves(history):

    # ---------------- Loss ----------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        linewidth=2,
        label="Training",
    )

    plt.plot(
        history.history["val_loss"],
        linewidth=2,
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

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

    # ---------------- Accuracy ----------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        linewidth=2,
        label="Training",
    )

    plt.plot(
        history.history["val_accuracy"],
        linewidth=2,
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

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

    # ---------------- Combined ----------------

    fig, ax1 = plt.subplots(figsize=(9,5))

    ax1.plot(
        history.history["loss"],
        linewidth=2,
    )

    ax1.set_xlabel("Epoch")

    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()

    ax2.plot(
        history.history["accuracy"],
        linewidth=2,
    )

    ax2.set_ylabel("Accuracy")

    plt.title("Training Curves")

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

def evaluate_model(model):

    y_prob = model.predict(
        X_test,
        verbose=0,
    )

    y_prob = y_prob.flatten()

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
        ),

        "F1-score": f1_score(
            y_test,
            y_pred,
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob,
        ),

    }

    print("\n")

    print("=" * 60)

    print("Evaluation Results")

    print("=" * 60)

    for metric, value in metrics.items():

        print(f"{metric:12s}: {value:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    return metrics, y_pred, y_prob


# ==========================================================

def save_metrics(metrics):

    metrics_df = pd.DataFrame({

        "Metric": list(metrics.keys()),

        "Value": list(metrics.values()),

    })

    metrics_df.to_csv(

        os.path.join(
            RESULTS_DIR,
            "federated_metrics.csv",
        ),

        index=False,

    )


# ==========================================================

def save_predictions(y_pred, y_prob):

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
# Plot Confusion Matrix
# ==========================================================

def plot_confusion(y_pred):

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
# Plot ROC Curve
# ==========================================================

def plot_roc(y_prob):

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

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
# Plot Precision Recall Curve
# ==========================================================

def plot_precision_recall(y_prob):

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob,
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

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

def plot_metrics(metrics):

    metric_names = list(metrics.keys())

    metric_values = list(metrics.values())

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

def plot_probability_distribution(y_prob):

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
# Save Everything
# ==========================================================

def save_all_results(
    history,
    model,
):

    print("\nSaving results...")

    save_training_history(history)

    plot_training_curves(history)

    metrics, y_pred, y_prob = evaluate_model(model)

    save_metrics(metrics)

    save_predictions(
        y_pred,
        y_prob,
    )

    plot_confusion(y_pred)

    plot_roc(y_prob)

    plot_precision_recall(y_prob)

    plot_metrics(metrics)

    plot_probability_distribution(y_prob)

    model.save(

        os.path.join(
            MODEL_DIR,
            "global_model.keras",
        )

    )

    print("\nEverything saved successfully.")

    print(f"\nResults saved to:\n{BASE_DIR}")
    
    # ==========================================================
# Centralised Training
# ==========================================================

def run_centralised():

    print("\n" + "=" * 60)
    print("Running Centralised Training")
    print("=" * 60)

    model = create_model()

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=50,

        batch_size=BATCH_SIZE,

        class_weight=CLASS_WEIGHTS,

        callbacks=[early_stop],

        verbose=1,

    )

    save_all_results(

        history,

        model,

    )

    print("\nCentralised experiment completed.")
    
    # ==========================================================
# Centralised Training
# ==========================================================

def run_centralised():

    print("\n" + "=" * 60)
    print("Running Centralised Training")
    print("=" * 60)

    model = create_model()

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=50,

        batch_size=BATCH_SIZE,

        class_weight=CLASS_WEIGHTS,

        callbacks=[early_stop],

        verbose=1,

    )

    save_all_results(

        history,

        model,

    )

    print("\nCentralised experiment completed.")

    # ======================================================
    # Final Evaluation
    # ======================================================

    metrics, y_pred, y_prob = evaluate_model(

        global_model

    )

    save_metrics(

        metrics

    )

    save_predictions(

        y_pred,

        y_prob,

    )

    plot_confusion(

        y_pred

    )

    plot_roc(

        y_prob

    )

    plot_precision_recall(

        y_prob

    )

    plot_metrics(

        metrics

    )

    plot_probability_distribution(

        y_prob

    )

    # ======================================================
    # Save Global Model
    # ======================================================

    global_model.save(

        os.path.join(

            MODEL_DIR,

            "global_model.keras",

        )

    )

    print("\n")

    print("=" * 60)

    print("Simulated Federated Learning Completed")

    print("=" * 60)

    print(f"\nResults saved to:\n{BASE_DIR}")
    
    # ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("Baseline Experiments")
    print("=" * 60)

    print(f"Experiment : {EXPERIMENT}")

    print(f"Dataset    : {DATASET}")

    print()

    if EXPERIMENT == "centralised":

        run_centralised()


    else:

        raise ValueError(
            "Unknown experiment."
        )

    print("\n")
    print("=" * 60)
    print("Experiment Finished Successfully")
    print("=" * 60)

    print("\nGenerated files:\n")

    print(BASE_DIR)

    print()

    print("models/")

    print("results/")

    print("result_plots/")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
