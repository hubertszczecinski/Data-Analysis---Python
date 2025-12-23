import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
}

accuracies = {}

for name, model in models.items():
    if name == "RF":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    accuracies[name] = accuracy_score(y_test, y_pred)


plt.figure(figsize=(5, 4))
plt.bar(list(accuracies.keys()), list(accuracies.values()), color=["#4c72b0", "#dd8452", "#55a868"])
plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy")
plt.title("Model comparison on test set")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad5_model_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()


