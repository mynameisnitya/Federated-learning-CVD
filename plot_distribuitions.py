import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("exploration_plots", exist_ok=True)

# ==========================================================
# Load datasets
# ==========================================================

iid_hospital1 = pd.read_csv("datasets/iid_hospital1.csv")
iid_hospital2 = pd.read_csv("datasets/iid_hospital2.csv")
iid_hospital3 = pd.read_csv("datasets/iid_hospital3.csv")

noniid_hospital1 = pd.read_csv("datasets/noniid_hospital1.csv")
noniid_hospital2 = pd.read_csv("datasets/noniid_hospital2.csv")
noniid_hospital3 = pd.read_csv("datasets/noniid_hospital3.csv")

# ==========================================================
# IID Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(iid_hospital1["AGE"], bins=30, alpha=0.5, label="Hospital 1")
plt.hist(iid_hospital2["AGE"], bins=30, alpha=0.5, label="Hospital 2")
plt.hist(iid_hospital3["AGE"], bins=30, alpha=0.5, label="Hospital 3")

plt.xlabel("Standardized AGE")
plt.ylabel("Number of Patients")
plt.title("IID Hospital Age Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    "exploration_plots/iid_age_distribution.png",
    dpi=300
)

plt.close()

# ==========================================================
# Non-IID Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(noniid_hospital1["AGE"], bins=30, alpha=0.5, label="Hospital 1")
plt.hist(noniid_hospital2["AGE"], bins=30, alpha=0.5, label="Hospital 2")
plt.hist(noniid_hospital3["AGE"], bins=30, alpha=0.5, label="Hospital 3")

plt.xlabel("Standardized AGE")
plt.ylabel("Number of Patients")
plt.title("Non-IID Hospital Age Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    "exploration_plots/noniid_age_distribution.png",
    dpi=300
)

plt.close()

print("Plots saved to exploration_plots/")
