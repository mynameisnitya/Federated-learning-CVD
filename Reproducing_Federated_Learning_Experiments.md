# Reproducing the Federated Learning Experiments

## 1. Open a terminal

Navigate to the project directory.

``` bash
cd ~/federated_learning
```

Activate the virtual environment.

``` bash
source venv/bin/activate
```

------------------------------------------------------------------------

# 2. Train the Centralised Baseline

## IID Dataset

``` bash
python baseline_training.py --experiment centralised --dataset iid
```

Outputs are saved to:

``` text
experiments/centralised_iid/
```

## Non-IID Dataset

``` bash
python baseline_training.py --experiment centralised --dataset noniid
```

Outputs are saved to:

``` text
experiments/centralised_noniid/
```

------------------------------------------------------------------------

# 3. Train FedProx (Non-IID)

## Terminal 1 -- Server

``` bash
python server.py --dataset noniid --strategy fedprox
```

## Terminal 2

``` bash
python client.py --dataset datasets/noniid_hospital1.csv
```

## Terminal 3

``` bash
python client.py --dataset datasets/noniid_hospital2.csv
```

## Terminal 4

``` bash
python client.py --dataset datasets/noniid_hospital3.csv
```

After training:

``` bash
python plot_training.py --experiment fedprox_noniid
python evaluate.py --dataset noniid --experiment fedprox_noniid
```

------------------------------------------------------------------------

# 4. Train FedAvg (Non-IID)

## Terminal 1

``` bash
python server.py --dataset noniid --strategy fedavg
```

## Terminal 2

``` bash
python client.py --dataset datasets/noniid_hospital1.csv
```

## Terminal 3

``` bash
python client.py --dataset datasets/noniid_hospital2.csv
```

## Terminal 4

``` bash
python client.py --dataset datasets/noniid_hospital3.csv
```

After training:

``` bash
python plot_training.py --experiment fedavg_noniid
python evaluate.py --dataset noniid --experiment fedavg_noniid
```

------------------------------------------------------------------------

# 5. Train FedAvg (IID)

## Terminal 1

``` bash
python server.py --dataset iid --strategy fedavg
```

## Terminal 2

``` bash
python client.py --dataset datasets/iid_hospital1.csv
```

## Terminal 3

``` bash
python client.py --dataset datasets/iid_hospital2.csv
```

## Terminal 4

``` bash
python client.py --dataset datasets/iid_hospital3.csv
```

After training:

``` bash
python plot_training.py --experiment fedavg_iid
python evaluate.py --dataset iid --experiment fedavg_iid
```

------------------------------------------------------------------------

# 6. Compare All Experiments

``` bash
python compare_results.py
```

------------------------------------------------------------------------

# Expected Folder Structure

``` text
experiments/
├── centralised_iid/
├── centralised_noniid/
├── fedavg_iid/
├── fedavg_noniid/
└── fedprox_noniid/
```
