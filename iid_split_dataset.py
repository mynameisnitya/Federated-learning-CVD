import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "AGE",
    "GENDER",
    "GLUCOSE",
    "CHOLESTEROL",
    "HEART_RATE",
    "SYS_BP",
    "DIA_BP",
]

# ==========================================================
# Create folders
# ==========================================================

os.makedirs("datasets", exist_ok=True)

# ==========================================================
# Load dataset
# ==========================================================

df = pd.read_csv("datasets/final_cvd_dataset.csv")

print("=" * 60)
print("Original Dataset")
print("=" * 60)
print(df.shape)

# ==========================================================
# Global Train/Test Split
# ==========================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["CVD"],
    random_state=RANDOM_STATE,
)

train_df.to_csv(
    "datasets/iid_train_global.csv",
    index=False,
)

test_df.to_csv(
    "datasets/iid_test_global.csv",
    index=False,
)

print(f"\nTraining Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

# ==========================================================
# IID Random Split
# ==========================================================

print("\nCreating IID hospital datasets...")

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

split_size = len(train_df) // 3

hospital1 = train_df.iloc[:split_size].copy()

hospital2 = train_df.iloc[split_size:2 * split_size].copy()

hospital3 = train_df.iloc[2 * split_size:].copy()

hospital1.to_csv(
    "datasets/iid_hospital1.csv",
    index=False,
)

hospital2.to_csv(
    "datasets/iid_hospital2.csv",
    index=False,
)

hospital3.to_csv(
    "datasets/iid_hospital3.csv",
    index=False,
)

hospitals = [
    hospital1,
    hospital2,
    hospital3,
]

# ==========================================================
# Summary
# ==========================================================

print("\n")
print("=" * 60)
print("IID Hospital Summary")
print("=" * 60)

for i, hospital in enumerate(hospitals, start=1):

    print(f"\nHospital {i}")
    print("-" * 40)

    print(f"Patients : {len(hospital)}")

    print("\nAverage Feature Values")

    print(
        hospital[FEATURES]
        .mean()
        .round(3)
    )

    print("\nCVD Distribution")

    print(
        hospital["CVD"]
        .value_counts(normalize=True)
        .round(3)
    )

# ==========================================================
# Validation
# ==========================================================

print("\n")
print("=" * 60)

print(
    f"Total Hospital Samples : {sum(len(h) for h in hospitals)}"
)

print(
    f"Training Samples       : {len(train_df)}"
)

if sum(len(h) for h in hospitals) == len(train_df):
    print("✓ Validation Passed")
else:
    print("✗ Validation Failed")

print("=" * 60)

print("\nFinished.")
