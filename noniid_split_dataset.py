import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 3

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
    "datasets/noniid_train_global.csv",
    index=False,
)

test_df.to_csv(
    "datasets/noniid_test_global.csv",
    index=False,
)

print(f"\nTraining Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

# ==========================================================
# KMeans Clustering
# ==========================================================

print("\nCreating heterogeneous hospitals using KMeans...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
)

train_df["Cluster"] = kmeans.fit_predict(
    train_df[FEATURES]
)

# ==========================================================
# Save Hospitals
# ==========================================================

hospitals = []

for cluster in range(N_CLUSTERS):

    hospital = (
        train_df[train_df["Cluster"] == cluster]
        .drop(columns=["Cluster"])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    hospitals.append(hospital)

    hospital.to_csv(
        f"datasets/noniid_hospital{cluster+1}.csv",
        index=False,
    )

# ==========================================================
# Summary
# ==========================================================

print("\n")
print("=" * 60)
print("Hospital Summary")
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
# Check Totals
# ==========================================================

total = sum(len(h) for h in hospitals)

print("\n")
print("=" * 60)

print(f"Total Hospital Samples : {total}")
print(f"Training Samples       : {len(train_df)}")

if total == len(train_df):
    print("✓ Validation Passed")
else:
    print("✗ Validation Failed")

print("=" * 60)

print("\nFinished.")
