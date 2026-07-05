# Federated Learning for Cardiovascular Disease Prediction

## Reproducibility Guide

## 1. System Requirements

### Hardware

-   CPU: Quad-core processor or better
-   RAM: Minimum 8 GB (16 GB recommended)
-   Storage: At least 5 GB free space

### Software

-   Ubuntu 22.04 LTS (or compatible Linux)
-   Python 3.12
-   TensorFlow 2.x (CPU version)
-   Flower
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib

------------------------------------------------------------------------

## 2. Project Structure

``` text
federated_learning/
│
├── datasets/
├── experiments/
├── models/
├── baseline_training.py
├── server.py
├── client.py
├── evaluate.py
├── plot_training.py
├── compare_results.py
├── model.py
├── data_loader.py
├── training_utils.py
└── requirements.txt
```

------------------------------------------------------------------------

## 3. Installation

Clone the repository:

``` bash
git clone <repository-url>
cd federated_learning
```

Create a virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 4. Dataset Preparation

Place the following CSV files inside the `datasets/` directory:

``` text
iid_train_global.csv
iid_test_global.csv
iid_hospital1.csv
iid_hospital2.csv
iid_hospital3.csv

noniid_train_global.csv
noniid_test_global.csv
noniid_hospital1.csv
noniid_hospital2.csv
noniid_hospital3.csv
```

The IID and Non-IID hospital datasets are partitions of the same global
dataset. The centralised baseline uses the global training dataset,
while the federated experiments use the hospital partitions.

------------------------------------------------------------------------

## 5. Running the Experiments

### Centralised Baseline

``` bash
python baseline_training.py --experiment centralised --dataset iid
python baseline_training.py --experiment centralised --dataset noniid
```

### FedProx (Non-IID)

Terminal 1:

``` bash
python server.py --dataset noniid --strategy fedprox
```

Terminal 2:

``` bash
python client.py --dataset datasets/noniid_hospital1.csv
```

Terminal 3:

``` bash
python client.py --dataset datasets/noniid_hospital2.csv
```

Terminal 4:

``` bash
python client.py --dataset datasets/noniid_hospital3.csv
```

After training:

``` bash
python plot_training.py --experiment fedprox_noniid
python evaluate.py --dataset noniid --experiment fedprox_noniid
```

### FedAvg (Non-IID)

``` bash
python server.py --dataset noniid --strategy fedavg
```

Run the three Non-IID hospital clients in separate terminals, then:

``` bash
python plot_training.py --experiment fedavg_noniid
python evaluate.py --dataset noniid --experiment fedavg_noniid
```

### FedAvg (IID)

``` bash
python server.py --dataset iid --strategy fedavg
```

Run the three IID hospital clients in separate terminals, then:

``` bash
python plot_training.py --experiment fedavg_iid
python evaluate.py --dataset iid --experiment fedavg_iid
```

------------------------------------------------------------------------

## 6. Compare Results

``` bash
python compare_results.py
```

------------------------------------------------------------------------

## 7. Outputs

Each experiment generates:

``` text
experiments/<experiment_name>/

models/
    global_model.keras

results/
    training_history.csv
    federated_metrics.csv
    predictions.csv

result_plots/
    training_accuracy.png
    training_loss.png
    training_curves.png
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    metrics_bar_chart.png
    probability_distribution.png
```

------------------------------------------------------------------------

## 8. Approximate Runtime

  Experiment          Approximate Time
  ----------------- ------------------
  Centralised             2--5 minutes
  FedAvg IID             5--10 minutes
  FedAvg Non-IID         5--10 minutes
  FedProx Non-IID        5--10 minutes

Times depend on hardware.

------------------------------------------------------------------------

## 9. Reproducibility

-   Random seed fixed to 42.
-   Identical neural network architecture across all experiments.
-   Same optimiser, learning rate, batch size, and local epochs.
-   Same global train/test split for all experiments.
-   Only the client partitioning strategy (IID vs Non-IID) and
    aggregation algorithm (FedAvg vs FedProx) differ.

------------------------------------------------------------------------

## 10. Citation

If using this implementation in academic work, please cite the
accompanying MSc dissertation.
