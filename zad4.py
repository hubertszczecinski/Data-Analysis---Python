import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
data_train = pd.read_csv("data3_train.csv", header=None, names=column_names)

X_train = data_train.iloc[:, :-1].values
y_train = data_train["species"].values


baseline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

baseline.fit(X_train, y_train)
pipeline.fit(X_train, y_train)

baseline_acc = baseline.score(X_train, y_train)
pipeline_acc = pipeline.score(X_train, y_train)


clf = pipeline.named_steps["clf"]
poly = pipeline.named_steps["poly"]
feature_names = poly.get_feature_names_out(column_names[:-1])
coef_abs = np.abs(clf.coef_).mean(axis=0)

base_features = column_names[:-1]
importance = {name: 0.0 for name in base_features}

for name, value in zip(feature_names, coef_abs):
    for base in base_features:
        if name.startswith(base):
            importance[base] += value

plt.figure(figsize=(6, 4))
plt.bar(list(importance.keys()), list(importance.values()), color="#4c72b0")
plt.ylabel("Aggregated importance")
plt.title("Feature importance from polynomial logistic regression")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad4_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(4, 4))
models = ["Baseline", "Pipeline"]
scores = [baseline_acc, pipeline_acc]
plt.bar(models, scores, color=["#4c72b0", "#dd8452"])
plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy (train)")
plt.title("Baseline vs engineered pipeline")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "zad4_pipeline_performance.png"), dpi=300, bbox_inches="tight")
plt.close()


