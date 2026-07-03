import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# =====================================================
# Features used throughout the analysis
# =====================================================

FEATURES = [
    "AGE",
    "GENDER",
    "GLUCOSE",
    "CHOLESTEROL",
    "HEART_RATE",
    "SYS_BP",
    "DIA_BP",
]


# =====================================================
# Verification Function
# =====================================================

def verify_dataset(DATASET_TYPE):

    print("\n")
    print("=" * 80)
    print(f"VERIFYING {DATASET_TYPE.upper()} DATASET")
    print("=" * 80)

    OUTPUT_DIR = os.path.join(
        "exploration_plots",
        DATASET_TYPE,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Load Hospital Datasets
    # -------------------------------------------------

    hospital1 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital1.csv"
    )

    hospital2 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital2.csv"
    )

    hospital3 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital3.csv"
    )

    hospitals = {

        "Hospital 1": hospital1,

        "Hospital 2": hospital2,

        "Hospital 3": hospital3,

    }

    # -------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------

    print("\nSUMMARY STATISTICS")

    print("-" * 80)

    for name, df in hospitals.items():

        print(f"\n{name}")

        print("-" * 40)

        print(df[FEATURES].describe().round(3))

        print("\nCVD Distribution")

        print(
            df["CVD"]
            .value_counts(normalize=True)
            .round(3)
        )

        print("\nAverage Feature Values")

        print(
            df[FEATURES]
            .mean()
            .round(3)
        )

    # =====================================================
    # AGE BOXPLOT
    # =====================================================

    plt.figure(figsize=(8,6))

    plt.boxplot(

        [

            hospital1["AGE"],

            hospital2["AGE"],

            hospital3["AGE"],

        ],

        tick_labels=[

            "Hospital 1",

            "Hospital 2",

            "Hospital 3",

        ],

    )

    plt.title(

        f"{DATASET_TYPE.upper()} AGE Distribution",

        fontsize=14,

        fontweight="bold",

    )

    plt.ylabel("Standardized AGE")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "age_boxplot.png",

        ),

        dpi=600,

    )

    plt.close()
    import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# =====================================================
# Features used throughout the analysis
# =====================================================

FEATURES = [
    "AGE",
    "GENDER",
    "GLUCOSE",
    "CHOLESTEROL",
    "HEART_RATE",
    "SYS_BP",
    "DIA_BP",
]


# =====================================================
# Verification Function
# =====================================================

def verify_dataset(DATASET_TYPE):

    print("\n")
    print("=" * 80)
    print(f"VERIFYING {DATASET_TYPE.upper()} DATASET")
    print("=" * 80)

    OUTPUT_DIR = os.path.join(
        "exploration_plots",
        DATASET_TYPE,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Load Hospital Datasets
    # -------------------------------------------------

    hospital1 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital1.csv"
    )

    hospital2 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital2.csv"
    )

    hospital3 = pd.read_csv(
        f"datasets/{DATASET_TYPE}_hospital3.csv"
    )

    hospitals = {

        "Hospital 1": hospital1,

        "Hospital 2": hospital2,

        "Hospital 3": hospital3,

    }

    # -------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------

    print("\nSUMMARY STATISTICS")

    print("-" * 80)

    for name, df in hospitals.items():

        print(f"\n{name}")

        print("-" * 40)

        print(df[FEATURES].describe().round(3))

        print("\nCVD Distribution")

        print(
            df["CVD"]
            .value_counts(normalize=True)
            .round(3)
        )

        print("\nAverage Feature Values")

        print(
            df[FEATURES]
            .mean()
            .round(3)
        )

    # =====================================================
    # AGE BOXPLOT
    # =====================================================

    plt.figure(figsize=(8,6))

    plt.boxplot(

        [

            hospital1["AGE"],

            hospital2["AGE"],

            hospital3["AGE"],

        ],

        tick_labels=[

            "Hospital 1",

            "Hospital 2",

            "Hospital 3",

        ],

    )

    plt.title(

        f"{DATASET_TYPE.upper()} AGE Distribution",

        fontsize=14,

        fontweight="bold",

    )

    plt.ylabel("Standardized AGE")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "age_boxplot.png",

        ),

        dpi=600,

    )

    plt.close()
    
    
    # =====================================================
    # Feature Boxplots
    # =====================================================

    print("\nGenerating feature boxplots...")

    for feature in FEATURES:

        plt.figure(figsize=(8,6))

        plt.boxplot(

            [

                hospital1[feature],

                hospital2[feature],

                hospital3[feature],

            ],

            tick_labels=[

                "Hospital 1",

                "Hospital 2",

                "Hospital 3",

            ],

        )

        plt.title(

            f"{DATASET_TYPE.upper()} {feature} Distribution",

            fontsize=14,

            fontweight="bold",

        )

        plt.ylabel(feature)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                OUTPUT_DIR,

                f"{feature.lower()}_boxplot.png",

            ),

            dpi=600,

        )

        plt.close()

    # =====================================================
    # CVD Prevalence
    # =====================================================

    print("Generating CVD prevalence plot...")

    prevalence = [

        hospital1["CVD"].mean(),

        hospital2["CVD"].mean(),

        hospital3["CVD"].mean(),

    ]

    plt.figure(figsize=(7,5))

    plt.bar(

        [

            "Hospital 1",

            "Hospital 2",

            "Hospital 3",

        ],

        prevalence,

    )

    plt.ylabel("Positive CVD Ratio")

    plt.ylim(0,1)

    plt.title(

        f"{DATASET_TYPE.upper()} CVD Prevalence",

        fontsize=14,

        fontweight="bold",

    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "cvd_prevalence.png",

        ),

        dpi=600,

    )

    plt.close()

    # =====================================================
    # Hospital Sizes
    # =====================================================

    print("Generating hospital size plot...")

    sizes = [

        len(hospital1),

        len(hospital2),

        len(hospital3),

    ]

    plt.figure(figsize=(7,5))

    plt.bar(

        [

            "Hospital 1",

            "Hospital 2",

            "Hospital 3",

        ],

        sizes,

    )

    plt.ylabel("Number of Patients")

    plt.title(

        f"{DATASET_TYPE.upper()} Hospital Sizes",

        fontsize=14,

        fontweight="bold",

    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "hospital_sizes.png",

        ),

        dpi=600,

    )

    plt.close()
    
    # =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("FEDERATED LEARNING DATASET VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------
    # IID Dataset
    # -------------------------------------------------

    verify_dataset("iid")

    print("\n")

    # -------------------------------------------------
    # Non-IID Dataset
    # -------------------------------------------------

    verify_dataset("noniid")

    print("\n")
    print("=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated folders:")

    print("exploration_plots/")

    print("   ├── iid/")
    print("   └── noniid/")

    print("\nEach folder contains:")

    print(" • age_boxplot.png")

    for feature in FEATURES:

        print(f" • {feature.lower()}_boxplot.png")

    print(" • cvd_prevalence.png")

    print(" • hospital_sizes.png")

    print(" • hospital1_correlation.png")
    print(" • hospital2_correlation.png")
    print(" • hospital3_correlation.png")

    print(" • hospital_pca.png")

    print("\nDone.")
