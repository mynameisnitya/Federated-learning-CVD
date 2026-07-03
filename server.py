import os
import argparse
import numpy as np
import pandas as pd
import flwr as fl

from model import create_model

# =====================================================
# Configuration
# =====================================================

NUM_ROUNDS = 20
SERVER_ADDRESS = "0.0.0.0:8080"

# =====================================================
# Command Line Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    choices=["iid", "noniid"],
    required=True,
)

parser.add_argument(
    "--strategy",
    choices=["fedavg", "fedprox"],
    required=True,
)

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy

EXPERIMENT = f"{STRATEGY}_{DATASET}"

# =====================================================
# Output folders
# =====================================================

MODEL_DIR = os.path.join(
    "experiments",
    EXPERIMENT,
    "models",
)

RESULTS_DIR = os.path.join(
    "experiments",
    EXPERIMENT,
    "results",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# Create model
# =====================================================

model = create_model()

training_history = []

# =====================================================
# Select Base Strategy
# =====================================================

if STRATEGY == "fedavg":

    BaseStrategy = fl.server.strategy.FedAvg

elif STRATEGY == "fedprox":

    BaseStrategy = fl.server.strategy.FedProx

else:

    raise ValueError("Unknown strategy")

# =====================================================
# Custom Strategy
# =====================================================

class SaveModelStrategy(BaseStrategy):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:

            model.set_weights(
                fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )
            )

            model.save(
                os.path.join(
                    MODEL_DIR,
                    "global_model.keras",
                )
            )

            loss = np.mean([
                fit_res.metrics["loss"]
                for _, fit_res in results
            ])

            accuracy = np.mean([
                fit_res.metrics["accuracy"]
                for _, fit_res in results
            ])

            training_history.append({

                "Round": server_round,

                "Loss": loss,

                "Accuracy": accuracy,

            })

            pd.DataFrame(
                training_history
            ).to_csv(

                os.path.join(
                    RESULTS_DIR,
                    "training_history.csv",
                ),

                index=False,

            )

            print()

            print("="*60)

            print(f"Round {server_round}")

            print(f"Loss     : {loss:.4f}")

            print(f"Accuracy : {accuracy:.4f}")

            print(f"Model saved to {MODEL_DIR}")

            print("="*60)

        return aggregated_parameters, metrics

# =====================================================
# Build Strategy
# =====================================================

strategy_kwargs = dict(

    fraction_fit=1.0,

    fraction_evaluate=0.0,

    min_fit_clients=3,

    min_available_clients=3,

    min_evaluate_clients=0,

)

if STRATEGY == "fedprox":

    strategy_kwargs["proximal_mu"] = 0.1

strategy = SaveModelStrategy(
    **strategy_kwargs
)

# =====================================================
# Start Server
# =====================================================

if __name__ == "__main__":

    print("="*60)
    print("Flower Federated Learning Server")
    print("="*60)

    print(f"Dataset   : {DATASET}")
    print(f"Strategy  : {STRATEGY}")
    print(f"Experiment: {EXPERIMENT}")

    fl.server.start_server(

        server_address=SERVER_ADDRESS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS,
        ),

        strategy=strategy,

    )
