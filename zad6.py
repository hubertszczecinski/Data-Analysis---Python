import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
data_train = pd.read_csv("data3_train.csv", header=None, names=column_names)

X_train = data_train.iloc[:, :-1].values
y_train = data_train["species"].values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


param_grid = {"n_neighbors": list(range(1, 21))}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv, scoring="accuracy")
grid.fit(X_train_scaled, y_train)


means = grid.cv_results_["mean_test_score"]
params = grid.cv_results_["param_n_neighbors"].data

plt.figure(figsize=(6, 4))
plt.plot(params, means, marker="o", color="#4c72b0")
plt.xlabel("k")
plt.ylabel("Mean CV accuracy")
plt.title("k-NN cross-validation accuracy vs k")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad6_knn_cv_accuracy_vs_k.png"), dpi=300, bbox_inches="tight")
plt.close()


