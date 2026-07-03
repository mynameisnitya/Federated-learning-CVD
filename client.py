import argparse
import warnings

import flwr as fl
import numpy as np

from model import create_model
from data_loader import load_training_data

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# Parse command-line argument
# -------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to hospital CSV",
)

args = parser.parse_args()

# -------------------------------------------------------
# Load local hospital dataset
# -------------------------------------------------------

X_train, y_train = load_training_data(args.dataset)

# -------------------------------------------------------
# Build model
# -------------------------------------------------------

model = create_model()

# -------------------------------------------------------
# Flower Client
# -------------------------------------------------------


class CVDClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        # Receive global weights
        model.set_weights(parameters)

        # Train locally
        history = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            verbose=1,
        )

        # Return updated weights
        return (
            model.get_weights(),
            len(X_train),
            {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
            },
        )

    def evaluate(self, parameters, config):

        # We won't evaluate here
        return 0.0, len(X_train), {"accuracy": 0.0}


# -------------------------------------------------------
# Start client
# -------------------------------------------------------

if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CVDClient(),
    )
