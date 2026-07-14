# Federated Learning for Cardiovascular Disease Prediction

## Overview

This repository contains the implementation developed for the MSc Data Science dissertation:

**"A Comparative Evaluation of Federated Learning Strategies for Privacy-Preserving Cardiovascular Disease Prediction"**

The project investigates the performance of Centralised Learning, FedAvg, and FedProx under both IID and Non-IID data distributions using the Flower Federated Learning Framework.

---

## Repository Structure

```
datasets/                  Dataset files and client partitions
experiments/               Individual experiment outputs
comparison_results/        Comparative metrics and figures
exploration_plots/         Exploratory data analysis

baseline_training.py       Centralised learning baseline
server.py                  Flower server
client.py                  Flower client
model.py                   Neural network architecture
data_loader.py             Dataset loading
training_utils.py          Training utilities
evaluate.py                Model evaluation
compare_results.py         Performance comparison
plot_training.py           Training visualisation
plot_training_comparison.py Training comparison plots
requirements.txt           Python dependencies
```

---

## Requirements

- Ubuntu 22.04 LTS (or compatible Linux)
- Python 3.12
- TensorFlow 2.x
- Flower
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The experiments use the cardiovascular disease dataset partitioned into:

- IID client datasets
- Non-IID client datasets
- Global training and testing datasets

Place all dataset files inside the `datasets/` directory before running the experiments.

---

## Running the Experiments

### Centralised Learning

```bash
python baseline_training.py
```

### Federated Learning

1. Start the Flower server.

```bash
python server.py
```

2. Launch three clients in separate terminals.

```bash
python client.py
```

3. Evaluate the trained model.

```bash
python evaluate.py
```

4. Generate comparison figures.

```bash
python compare_results.py
python plot_training_comparison.py
```

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Clients | 3 |
| Random Seed | 42 |
| Local Epochs | 1 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimiser | Adam |
| Aggregation Methods | FedAvg, FedProx |

---

## Generated Outputs

The implementation automatically generates:

- Training accuracy and loss curves
- Classification metrics
- Accuracy, Precision, Recall, F1-score and ROC-AUC comparisons
- Comparative performance tables
- Experimental summary reports

Outputs are stored within the `experiments/` and `comparison_results/` directories.

---

## Reproducibility

All experiments were executed using identical:

- Neural network architecture
- Hyperparameters
- Optimisation settings
- Train/test split
- Random seed (42)

The only experimental variables are:

- Data partitioning strategy (IID vs Non-IID)
- Federated aggregation algorithm (FedAvg vs FedProx)

---

## Author

**Nitya Puspaceno**

MSc Data Science

Nottingham Trent University

Academic Year: 2025–2026
