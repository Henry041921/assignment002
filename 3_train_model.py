import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import math
from collections import Counter

from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_FILE = "landmarks_data.csv"
MODEL_SAVE_PATH = "gesture_model.pkl"

# Hand-coded KNN class without sklearn
# Logic: Calculate distance -> Find neighbors -> Vote.
class KNN_From_Scratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        # KNN is lazy learning, so we just store the data.
        # Convert everything to Python lists to be safe and use standard libs only.
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
        # Calculate Euclidean distance manually using math library.
        # It's just the square root of the sum of squared differences for each feature.
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    def predict(self, X):
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
        # 1. Calculate distance from this point to all training points.
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(row, self.X_train[i])
            distances.append((self.X_train[i], self.y_train[i], dist))

        # 2. Sort by distance to find the nearest ones.
        distances.sort(key=lambda tup: tup[2])

        # 3. Get the top k nearest neighbors.
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][1])

        # 4. Voting: Use Counter to find the most common label among neighbors.
        # This decides the final class.
        vote_result = Counter(neighbors).most_common(1)[0][0]
        return vote_result

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def run_cross_validation(model, X, y, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []

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
    print(f"{title} saved as image")

def plot_model_comparison(cv_scores, test_scores):
    data = []
    for model_name in cv_scores.keys():
        data.append({'Model': model_name, 'Accuracy': cv_scores[model_name], 'Type': 'CV Score (Train)'})
        data.append({'Model': model_name, 'Accuracy': test_scores[model_name], 'Type': 'Test Score'})

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(x='Model', y='Accuracy', hue='Type', data=df_plot, palette="viridis")

    plt.title("Model Performance Comparison: CV vs Test", fontsize=15)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.15)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if not math.isnan(height) and height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 0.015,
                    f'{height:.1%}', ha="center", va="bottom", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("model_comparison_double.png")
    print(f"Comparison plot saved as: model_comparison_double.png")

if __name__ == "__main__":
    print(f"Loading data from {DATA_FILE}...")
    try:
        # Load the CSV data generated by MediaPipe.
        # Note: MediaPipe already normalized the coordinates (0-1), which is crucial for KNN distance accuracy.
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Please run 2a_feature_extraction.py first.")
        exit()

    X = df.drop('label', axis=1)
    y = df['label']

    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cv_results = {}
    test_results = {}

    print("\n" + "=" * 50)
    print("Part 2c: Supervised Learning Optimization & Evaluation")
    print("=" * 50)

    # Model A: Decision Tree (Required)
    print("\n1. Optimizing Decision Tree...")
    best_dt_score = 0
    best_dt_depth = None

    # Parameter Tuning: Max Depth
    for depth in [3, 5, 10, 15, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        score = run_cross_validation(dt, X_train, y_train, k_folds=5)
        print(f"   Depth={str(depth):<4} | CV Accuracy: {score:.4f}")
        if score > best_dt_score:
            best_dt_score = score
            best_dt_depth = depth

    print(f"   Best Depth: {best_dt_depth}")

    final_dt = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
    final_dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, final_dt.predict(X_test))

    cv_results['Decision Tree'] = best_dt_score
    test_results['Decision Tree'] = dt_acc

    # Model B: Random Forest
    print("\n2. Optimizing Random Forest (Selected Model)...")
    best_rf_score = 0
    best_rf_est = None

    # Parameter Tuning: n_estimators (number of trees)
    for n_est in [10, 50, 100]:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        score = run_cross_validation(rf, X_train, y_train, k_folds=5)
        print(f"   Trees={str(n_est):<4} | CV Accuracy: {score:.4f}")
        if score > best_rf_score:
            best_rf_score = score
            best_rf_est = n_est

    print(f"   Best Trees: {best_rf_est}")

    final_rf = RandomForestClassifier(n_estimators=best_rf_est, random_state=42)
    final_rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, final_rf.predict(X_test))

    cv_results['Random Forest'] = best_rf_score
    test_results['Random Forest'] = rf_acc

    # Model C: Custom KNN
    print("\n3. Optimizing Custom KNN (From Scratch)...")
    best_knn_score = 0
    best_k = None

    #Parameter Tuning: K Value
    for k in [1, 3, 5, 7]:
        knn = KNN_From_Scratch(k=k)
        score = run_cross_validation(knn, X_train, y_train, k_folds=5)
        print(f"   k={str(k):<8} | CV Accuracy: {score:.4f}")
        if score > best_knn_score:
            best_knn_score = score
            best_k = k

    print(f"   Best k: {best_k}")

    final_knn = KNN_From_Scratch(k=best_k)
    final_knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, final_knn.predict(X_test))

    cv_results['KNN (Custom)'] = best_knn_score
    test_results['KNN (Custom)'] = knn_acc

    print("\n" + "=" * 50)
    print("Final Test Set Results")
    print("=" * 50)
    best_model_name = ""
    best_model_acc = 0

    for name, acc in test_results.items():
        print(f"{name:<20}: {acc:.2%}")
        if acc > best_model_acc:
            best_model_acc = acc
            best_model_name = name

    print("-" * 50)
    print(f"Best model is: {best_model_name}")

    plot_model_comparison(cv_results, test_results)

    # Draw a double bar chart for comparison.
    print("\nGenerating Confusion Matrix for the best model...")
    if best_model_name == 'Decision Tree':
        y_pred = final_dt.predict(X_test)
        save_model = final_dt
    elif best_model_name == 'Random Forest':
        y_pred = final_rf.predict(X_test)
        save_model = final_rf
    else:
        y_pred = final_knn.predict(X_test)
        save_model = final_knn

    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {best_model_name}")

    # save model
    joblib.dump(save_model, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("\nNext step: Run 4_realtime_recognition.py to see real-time usage!")