import pandas as pd
import numpy as np  # 仅用于数据加载和分割，不用于KNN核心逻辑
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import math
from collections import Counter

# 允许使用 sklearn 进行数据分割、评估和其他模型 (例外是 KNN)
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 配置 ---
DATA_FILE = "landmarks_data.csv"
MODEL_SAVE_PATH = "gesture_model.pkl"


# ==========================================
# 🌟 核心得分点：纯手写 KNN (From Scratch)
# ⚠️ 严格遵守文档 Part 2c 要求：
# "implemented from scratch using only Python standard built-in libraries"
# ==========================================
class KNN_From_Scratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        训练过程其实就是存储数据。
        为了符合"仅使用内置库"的要求，我们将数据转换为纯 Python list。
        """
        # 如果输入是 DataFrame 或 Numpy 数组，转换为 list
        if hasattr(X, 'values'):
            self.X_train = X.values.tolist()
        elif hasattr(X, 'tolist'):
            self.X_train = X.tolist()
        else:
            self.X_train = list(X)

        if hasattr(y, 'values'):
            self.y_train = y.values.tolist()
        elif hasattr(y, 'tolist'):
            self.y_train = y.tolist()
        else:
            self.y_train = list(y)

    def _euclidean_distance(self, row1, row2):
        """仅使用 math 库计算欧几里得距离"""
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    def predict(self, X):
        """预测新数据"""
        # 转换输入数据为 list
        if hasattr(X, 'values'):
            X_data = X.values.tolist()
        elif hasattr(X, 'tolist'):
            X_data = X.tolist()
        else:
            X_data = list(X)

        predictions = []
        for row in X_data:
            label = self._predict_single(row)
            predictions.append(label)
        return predictions

    def _predict_single(self, row):
        # 1. 计算距离
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(row, self.X_train[i])
            distances.append((self.X_train[i], self.y_train[i], dist))

        # 2. 按距离排序 (从小到大)
        distances.sort(key=lambda tup: tup[2])

        # 3. 获取最近的 k 个邻居
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][1])  # 只取标签

        # 4. 投票 (使用 collections.Counter)
        vote_result = Counter(neighbors).most_common(1)[0][0]
        return vote_result

    # 为了兼容 sklearn 的接口 (cross_val_score 需要这个)
    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# ==========================================
# 🛠️ 辅助功能
# ==========================================
def run_cross_validation(model, X, y, k_folds=5):
    """执行 5-Fold Cross Validation 并返回平均准确率"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []

    # 转换为 numpy 以方便索引切片
    X_np = np.array(X)
    y_np = np.array(y)

    for train_idx, val_idx in kf.split(X_np):
        X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
        y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, preds)
        scores.append(score)

    return np.mean(scores)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{title.replace(' ', '_')}.png")
    print(f"📊 {title} 已保存为图片")


# ==========================================
# 🚀 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ 错误：找不到 CSV 文件。请先运行 2a_feature_extraction.py")
        exit()

    X = df.drop('label', axis=1)
    y = df['label']

    # 2. 划分数据集 (80% 训练, 20% 测试) [cite: 168]
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}  # 存储最终结果

    print("\n" + "=" * 50)
    print("🤖 Part 2c: 监督学习优化与评估")
    print("=" * 50)

    # --- 模型 A: Decision Tree (必选) ---
    print("\n🌲 1. Optimizing Decision Tree...")
    best_dt_score = 0
    best_dt_depth = None

    # 调参: Max Depth
    for depth in [3, 5, 10, 15, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        score = run_cross_validation(dt, X_train, y_train, k_folds=5)
        print(f"   Depth={str(depth):<4} | CV Accuracy: {score:.4f}")
        if score > best_dt_score:
            best_dt_score = score
            best_dt_depth = depth

    print(f"   ✅ Best Depth: {best_dt_depth}")

    # 用最佳参数在完整训练集上重训，并在测试集上评估 [cite: 175-177]
    final_dt = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
    final_dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, final_dt.predict(X_test))
    results['Decision Tree'] = dt_acc

    # --- 模型 B: Random Forest (自选模型) ---
    print("\n🌳 2. Optimizing Random Forest (Selected Model)...")
    best_rf_score = 0
    best_rf_est = None

    # 调参: n_estimators (树的数量)
    for n_est in [10, 50, 100]:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        score = run_cross_validation(rf, X_train, y_train, k_folds=5)
        print(f"   Trees={str(n_est):<4} | CV Accuracy: {score:.4f}")
        if score > best_rf_score:
            best_rf_score = score
            best_rf_est = n_est

    print(f"   ✅ Best Trees: {best_rf_est}")

    final_rf = RandomForestClassifier(n_estimators=best_rf_est, random_state=42)
    final_rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, final_rf.predict(X_test))
    results['Random Forest'] = rf_acc

    # --- 模型 C: Custom KNN (From Scratch 必选) ---
    print("\n🤝 3. Optimizing Custom KNN (From Scratch)...")
    best_knn_score = 0
    best_k = None

    # 调参: K Value
    for k in [1, 3, 5, 7]:
        knn = KNN_From_Scratch(k=k)
        score = run_cross_validation(knn, X_train, y_train, k_folds=5)
        print(f"   k={str(k):<8} | CV Accuracy: {score:.4f}")
        if score > best_knn_score:
            best_knn_score = score
            best_k = k

    print(f"   ✅ Best k: {best_k}")

    final_knn = KNN_From_Scratch(k=best_k)
    final_knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, final_knn.predict(X_test))
    results['KNN (Custom)'] = knn_acc

    # --- 总结与保存 ---
    print("\n" + "=" * 50)
    print("🏆 Final Test Set Results")
    print("=" * 50)
    best_model_name = ""
    best_model_acc = 0

    for name, acc in results.items():
        print(f"{name:<20}: {acc:.2%}")
        if acc > best_model_acc:
            best_model_acc = acc
            best_model_name = name

    print("-" * 50)
    print(f"🌟 最佳模型是: {best_model_name}")

    # 画出最佳模型的混淆矩阵
    print("Generating Confusion Matrix for the best model...")
    if best_model_name == 'Decision Tree':
        y_pred = final_dt.predict(X_test)
        save_model = final_dt
    elif best_model_name == 'Random Forest':
        y_pred = final_rf.predict(X_test)
        save_model = final_rf
    else:
        y_pred = final_knn.predict(X_test)
        # joblib 保存自定义类可能会有兼容性问题，通常推荐保存 RF
        # 但如果 KNN 最好，我们还是尝试保存它
        save_model = final_knn

    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {best_model_name}")

    # 保存模型
    joblib.dump(save_model, MODEL_SAVE_PATH)
    print(f"💾 模型已保存至: {MODEL_SAVE_PATH}")
    print("\n下一步：请运行 4_realtime_recognition.py 查看实时效果！")