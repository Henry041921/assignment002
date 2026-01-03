import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score, confusion_matrix

# --- 配置 ---
DATA_FILE = "landmarks_data.csv"


def plot_cluster_comparison(y_true, y_cluster, title):
    """画图对比：横轴是聚类结果，纵轴是真实标签"""
    # 创建一个透视表来看每个Cluster里主要包含了哪些真实标签
    df_cm = pd.DataFrame({'True Label': y_true, 'Cluster ID': y_cluster})
    ct = pd.crosstab(df_cm['True Label'], df_cm['Cluster ID'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.ylabel('True Label (Ground Truth)')
    plt.xlabel('Cluster ID (AI Found)')
    plt.tight_layout()
    plt.savefig(f"clustering_{title.replace(' ', '_')}.png")
    print(f"📊 图表已保存: clustering_{title.replace(' ', '_')}.png")


if __name__ == "__main__":
    print("Loading data for Clustering...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ 找不到 CSV 文件")
        exit()

    # 1. 准备数据 (去掉标签) [cite: 182]
    X = df.drop('label', axis=1)
    y_true = df['label']  # 保留标签只是为了验证效果，训练时不给模型看

    # 既然只有 A-J 共 10 个手势，我们设 K=10
    n_clusters = 10
    print(f"\n🤖 开始聚类 (Target Clusters = {n_clusters})")

    # --- A. K-Means Clustering [cite: 184] ---
    print("\n🔹 Running K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # 评估 K-Means
    # ARI (Adjusted Rand Index): 0=随机, 1=完美匹配
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    # Homogeneity: 每个簇是否只包含单一类别的样本
    homo_kmeans = homogeneity_score(y_true, kmeans_labels)

    print(f"   K-Means ARI Score: {ari_kmeans:.4f} (越接近1越好)")
    print(f"   K-Means Homogeneity: {homo_kmeans:.4f}")
    plot_cluster_comparison(y_true, kmeans_labels, "K-Means Clustering Analysis")

    # --- B. Hierarchical Clustering (层次聚类) [cite: 184] ---
    print("\n🔹 Running Hierarchical Clustering...")
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    hc_labels = hc.fit_predict(X)

    ari_hc = adjusted_rand_score(y_true, hc_labels)
    homo_hc = homogeneity_score(y_true, hc_labels)

    print(f"   Hierarchical ARI Score: {ari_hc:.4f}")
    print(f"   Hierarchical Homogeneity: {homo_hc:.4f}")
    plot_cluster_comparison(y_true, hc_labels, "Hierarchical Clustering Analysis")

    # --- C. 总结与对比分析 [cite: 186] ---
    print("\n" + "=" * 40)
    print("📝 聚类结果分析 (Clustering Analysis)")
    print("=" * 40)
    print("你应该在报告里写：")
    if ari_kmeans > 0.5:
        print("✅ K-Means 效果不错，能够大致区分出不同的手势。")
        print("   这证明了我们的特征提取(Pre-processing)非常有效，")
        print("   即使没有标签，数据在空间里也是自然分开的。")
    else:
        print("⚠️ 聚类效果一般。说明某些手势长得很像 (比如 A, M, N 可能容易混)，")
        print("   在没有标签监督的情况下，AI 很难把它们区分开。")

    print("-" * 40)
    print("现在，Part 2d (Unsupervised Learning) 任务完成！")