import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score, confusion_matrix

DATA_FILE = "landmarks_data.csv"

def plot_cluster_comparison(y_true, y_cluster, title):
    # Create a crosstab to see how clusters map to true labels
    df_cm = pd.DataFrame({'True Label': y_true, 'Cluster ID': y_cluster})
    ct = pd.crosstab(df_cm['True Label'], df_cm['Cluster ID'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.ylabel('True Label (Ground Truth)')
    plt.xlabel('Cluster ID (AI Found)')
    plt.tight_layout()
    plt.savefig(f"clustering_{title.replace(' ', '_')}.png")
    print(f"Plot saved: clustering_{title.replace(' ', '_')}.png")

if __name__ == "__main__":
    print("Loading data for Clustering...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("CSV file not found")
        exit()

    # Separate features and labels for unsupervised learning
    X = df.drop('label', axis=1)
    y_true = df['label']

    n_clusters = 10
    print(f"\nStarting Clustering (Target Clusters = {n_clusters})")

    # --- K-Means Clustering ---
    print("\nRunning K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # Evaluate K-Means using ARI and Homogeneity scores
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    homo_kmeans = homogeneity_score(y_true, kmeans_labels)

    print(f"   K-Means ARI Score: {ari_kmeans:.4f} (Closer to 1 is better)")
    print(f"   K-Means Homogeneity: {homo_kmeans:.4f}")
    plot_cluster_comparison(y_true, kmeans_labels, "K-Means Clustering Analysis")

    # --- Hierarchical Clustering ---
    print("\nRunning Hierarchical Clustering...")
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    hc_labels = hc.fit_predict(X)

    ari_hc = adjusted_rand_score(y_true, hc_labels)
    homo_hc = homogeneity_score(y_true, hc_labels)

    print(f"   Hierarchical ARI Score: {ari_hc:.4f}")
    print(f"   Hierarchical Homogeneity: {homo_hc:.4f}")
    plot_cluster_comparison(y_true, hc_labels, "Hierarchical Clustering Analysis")

    # --- Analysis Summary ---
    print("\n" + "=" * 40)
    print("Clustering Analysis Result")
    print("=" * 40)
    print("Report summary:")
    if ari_kmeans > 0.5:
        print("K-Means worked well, roughly separating different gestures.")
        print("This proves our feature extraction is effective.")
        print("The data is naturally separable in space even without labels.")
    else:
        print("Clustering performance is average. Some gestures look similar.")
        print("Without supervision, the AI struggles to distinguish them (e.g., A, M, N).")

    print("-" * 40)
    print("Part 2d (Unsupervised Learning) task completed!")