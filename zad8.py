import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
data_train_full = pd.read_csv("data3_train.csv", header=None, names=column_names)
data_test = pd.read_csv("data3_test.csv", header=None, names=column_names)

X_test = data_test.iloc[:, :-1].values
y_test = data_test["species"].values


majority_class = data_train_full["species"].value_counts().idxmax()
minority_classes = [c for c in data_train_full["species"].unique() if c != majority_class]

frames = [data_train_full[data_train_full["species"] == majority_class]]
for c in minority_classes:
    subset = data_train_full[data_train_full["species"] == c]
    frac = 0.3
    frames.append(subset.sample(frac=frac, random_state=42))

data_train_imbalanced = pd.concat(frames, ignore_index=True)


data_train_balanced = data_train_full.copy()


def train_and_eval(train_data):
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data["species"].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return labels, per_class_acc


labels_bal, acc_bal = train_and_eval(data_train_balanced)
labels_imb, acc_imb = train_and_eval(data_train_imbalanced)


labels = labels_bal
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width / 2, acc_bal, width, label="Balanced", color="#4c72b0")
plt.bar(x + width / 2, acc_imb, width, label="Imbalanced", color="#dd8452")
plt.xticks(x, labels, rotation=0)
plt.ylim(0.0, 1.0)
plt.ylabel("Per-class accuracy")
plt.title("Impact of class imbalance on per-class accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad8_class_accuracy_balanced_vs_imbalanced.png"), dpi=300, bbox_inches="tight")
plt.close()


