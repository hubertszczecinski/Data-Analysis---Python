import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
data = pd.read_csv("data3_train.csv", header=None, names=column_names)

X = data.iloc[:, :-1].values
y = data["species"].values


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ]
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")


pipeline.fit(X_train, y_train)
val_accuracy = pipeline.score(X_val, y_val)


plt.figure(figsize=(5, 4))
fold_ids = np.arange(1, len(cv_scores) + 1)
plt.bar(fold_ids, cv_scores, color="#4c72b0")
plt.ylim(0.0, 1.0)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-validation accuracy per fold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad10_cv_fold_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(4, 4))
values = list(cv_scores) + [val_accuracy]
labels = [f"CV {i}" for i in fold_ids] + ["Hold-out"]
positions = np.arange(len(values))
plt.bar(positions, values, color=["#4c72b0"] * len(cv_scores) + ["#dd8452"])
plt.xticks(positions, labels, rotation=45, ha="right")
plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy")
plt.title("End-to-end pipeline performance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad10_pipeline_performance.png"), dpi=300, bbox_inches="tight")
plt.close()


