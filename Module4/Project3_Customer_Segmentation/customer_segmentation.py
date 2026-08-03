"""Project 3 - Customer Segmentation (Unsupervised - Clustering).

Groups customers into segments with NO labels using K-Means on two
features: annual income (in k) and spending score.

Per Module 4 notes, section 14:
- X = df[["AnnualIncome_k", "SpendingScore"]]
- scale first (StandardScaler) - K-Means is distance-based
- Elbow method (k = 1..10) to sanity-check k, then fit with k=5
- KMeans(n_clusters=5, random_state=42, n_init=10)
- describe each segment and NAME it with a business label

The script GENERATES its own synthetic dataset (200 customers = 5 blobs
of 40, exactly like the sample output) with random_state=42 so every run
is reproducible. It saves one PNG with the elbow plot next to the colored
clusters with their centroids.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def make_synthetic_customers(n=200, seed=42):
    """Generate 5 customer blobs (40 each) - income in k, spend 0-100."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [78.4, 19.6],   # high income, low spend
        [30.6, 79.1],   # low income, high spend
        [85.6, 82.4],   # high income, high spend
        [30.3, 30.1],   # low income, low spend
        [55.5, 50.2],   # middle of both
    ])
    scale = np.array([[7.0, 7.0]])
    blobs = np.vstack([centers[i] + rng.normal(0, 1, (40, 2)) * scale
                       for i in range(centers.shape[0])])
    return pd.DataFrame(blobs, columns=["AnnualIncome_k", "SpendingScore"])


def name_segment(income, spend):
    """Give each cluster a business name based on its feature averages."""
    if income < 40 and spend < 45:
        return "Budget"
    if income < 40 and spend >= 60:
        return "Young Spenders"
    if 42 <= income <= 68:
        return "Average"
    if income > 70 and spend < 40:
        return "Savers (win them over)"
    if income > 70 and spend >= 60:
        return "Premium (VIP - target!)"
    return "Average"


def main():
    print("=" * 60)
    print("PROJECT 3 - CUSTOMER SEGMENTATION (CLUSTERING)")
    print("=" * 60)

    # 1. Prepare - 2 features, no label y (unsupervised!)
    df = make_synthetic_customers()
    print(f"Generated synthetic dataset: {df.shape[0]} customers "
          f"(features: {list(df.columns)}) - NO labels needed")

    X = df[["AnnualIncome_k", "SpendingScore"]].values
    X_scaled = StandardScaler().fit_transform(X)  # scale first!

    # 2. Elbow method to sanity-check k
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10)
                .fit(X_scaled).inertia_ for k in range(1, 11)]
    print()
    print("----- ELBOW METHOD (inertia for k = 1..10) -----")
    print("   k : inertia")
    for k, inert in enumerate(inertias, start=1):
        print(f"   {k:2d} : {inert:9.1f}")
    print("   The curve flattens sharply after k=5 -> elbow at k=5")

    # 3. Fit K-Means with the chosen k
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Segment"] = kmeans.fit_predict(X_scaled)  # no y - unsupervised

    # 4. Describe each segment and NAME it
    print()
    print("----- CUSTOMER SEGMENTS FOUND -----")
    seg_means = df.groupby("Segment")[["AnnualIncome_k", "SpendingScore"]].mean()
    for seg in sorted(seg_means.index):
        row = seg_means.loc[seg]
        count = (df["Segment"] == seg).sum()
        name = name_segment(row["AnnualIncome_k"], row["SpendingScore"])
        print(f"   Segment {seg}: {count:3d} customers | "
              f"avg income {row['AnnualIncome_k']:6.1f}k | "
              f"avg spend {row['SpendingScore']:5.1f} | {name}")

    print()
    print("The algorithm discovered 5 clean segments with no labels at all -")
    print("then we gave each one a business name.")

    # 5. Save elbow plot next to the colored clusters with centroids
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(1, 11), inertias, marker="o")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("inertia (within-cluster distance)")
    ax1.set_title("Elbow Method - pick k at the elbow")
    ax1.axvline(5, color="r", linestyle="--", label="chosen k=5")
    ax1.legend()

    centers_scaled = kmeans.cluster_centers_
    # Map centroids back to the original scale for plotting
    scaler = StandardScaler()
    scaler.fit(X)
    centers_orig = scaler.inverse_transform(centers_scaled)
    sc = ax2.scatter(X[:, 0], X[:, 1], c=df["Segment"], cmap="tab10",
                     alpha=0.7, s=40)
    ax2.scatter(centers_orig[:, 0], centers_orig[:, 1], marker="X",
                c="red", s=200, edgecolors="k", label="centroids")
    ax2.set_xlabel("Annual income (k)")
    ax2.set_ylabel("Spending score")
    ax2.set_title("Customer segments (k=5) with centroids")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("segments_and_elbow.png", dpi=120)
    print()
    print("Saved chart: segments_and_elbow.png")


if __name__ == "__main__":
    main()
