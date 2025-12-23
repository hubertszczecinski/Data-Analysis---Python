import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
data_train = pd.read_csv("data3_train.csv", header=None, names=column_names)
data_test = pd.read_csv("data3_test.csv", header=None, names=column_names)

X_train = data_train.iloc[:, :-1].values
y_train = data_train["species"].values
X_test = data_test.iloc[:, :-1].values
y_test = data_test["species"].values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)


report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
labels = sorted(set(y_test))
f1_scores = [report[str(label)]["f1-score"] for label in labels]

plt.figure(figsize=(5, 4))
plt.bar(labels, f1_scores, color="#4c72b0")
plt.ylim(0.0, 1.0)
plt.ylabel("F1-score")
plt.title("Per-class F1-score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad7_f1_per_class.png"), dpi=300, bbox_inches="tight")
plt.close()


cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(4.5, 4))
im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.colorbar(im, fraction=0.046, pad=0.04)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix (k-NN)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad7_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()


