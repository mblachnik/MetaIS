import pandas as pd
import matplotlib.pyplot as plt

# dane (bez kolumny #samples)
data = {
    "dataset": [
        "spambase", "banana", "phoneme", "page-blocks", "texture", "optdigits", 
        "satimage", "marketing", "ring", "twonorm", "penbased", "nursery", 
        "magic", "letter", "electricity-norm.", "shuttle", "codrnaNorm", "covtype"
    ],
    "HMN-EI": [
        1.67, 0.56, 0.62, 0.95, 2.11, 2.98, 2.31, 1.47, 2.90, 1.61,
        3.12, 4.28, 1.57, 6.57, 3.08, 7.57, 13.37, 18.47
    ],
    "CCIS": [
        1.35, 0.42, 0.59, 0.97, 2.15, 2.74, 2.44, 1.28, 1.29, 1.09,
        2.80, 3.23, 1.51, 11.69, 4.46, 4.85, 7.06, 37.97
    ],
    "ENN": [
        2.75, 0.50, 0.61, 0.91, 2.50, 3.53, 2.48, 1.25, 2.02, 1.52,
        2.66, 5.00, 2.49, 3.26, 4.49, 8.29, 22.98, 39.11
    ],
    "ICF": [
        2.24, 0.51, 0.58, 0.90, 2.30, 3.10, 2.10, 1.01, 1.42, 1.74,
        2.27, 5.36, 2.37, 3.71, 4.88, 6.46, 11.92, 69.26
    ],
    "Drop3": [
        14.09, 7.63, 7.08, 14.65, 15.40, 20.17, 16.68, 1.55, 10.46, 7.20,
        21.05, 36.94, 15.64, 33.37, 34.90, 214.94, None, None  # brak wartości dla 2 ostatnich
    ]
}

df = pd.DataFrame(data)

# lista algorytmów
methods = ["HMN-EI", "CCIS", "ENN", "ICF", "Drop3"]

# wykresy słupkowe
for method in methods:
    plt.figure(figsize=(12,6))
    plt.bar(df["dataset"], df[method], color="skyblue", edgecolor="black")
    plt.xticks(rotation=75, ha="right", fontsize=14)
    plt.ylabel("Speedup", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.gcf().savefig(f"Y:\\MetaIS\\results\\figs\\speedup_{method}.png", dpi=300)
