import pandas as pd

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


def load_training_data(csv_file):

    df = pd.read_csv(csv_file)

    X = df[FEATURES].values.astype("float32")
    y = df[TARGET].values.astype("float32")

    return X, y


def load_test_data(dataset_type):

    csv_file = f"datasets/{dataset_type}_test_global.csv"

    df = pd.read_csv(csv_file)

    X = df[FEATURES].values.astype("float32")
    y = df[TARGET].values.astype("float32")

    return X, y
