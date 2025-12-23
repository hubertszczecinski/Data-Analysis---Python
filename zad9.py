import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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


rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_scaled, y_train)


importances = rf.feature_importances_
features = column_names[:-1]

plt.figure(figsize=(5, 4))
order = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in order]
sorted_importances = importances[order]
plt.bar(sorted_features, sorted_importances, color="#4c72b0")
plt.ylabel("Importance")
plt.title("Random forest feature importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad9_feature_importance_random_forest.png"), dpi=300, bbox_inches="tight")
plt.close()


