## Data-Analysis---Python

This repository contains a small end‑to‑end analytics project built on the **Iris** dataset.  
Implemented a complete machine learning workflow: data analysis, clustering, supervised learning, feature engineering, model comparison, cross-validation, handling imbalanced datasets, and production-ready inference pipelines.  
The Iris dataset here is a small, canonical benchmark used to demonstrate the full workflow, not a realistic production case.


### Project structure

- **`zad1.py`** – exploratory data analysis (EDA) and basic statistics.
- **`zad2.py`** – KMeans clustering on normalized features.
- **`zad3.py`** – k‑NN classification, model evaluation and confusion matrices.
- **`zad4.py`** – feature engineering and simple pipelines.
- **`zad5.py`** – model comparison across several algorithms.
- **`zad6.py`** – cross-validation and basic hyperparameter tuning.
- **`zad7.py`** – evaluation metrics and diagnostic plots.
- **`zad8.py`** – experiments with imbalanced data.
- **`zad9.py`** – model interpretability with feature importance.
- **`zad10.py`** – end-to-end ML pipeline with validation.
- **`data1.csv`, `data2.csv`, `data3_train.csv`, `data3_test.csv`** – input data files.
- **`plots/`** – directory with generated figures (PNG).


### 1. Exploratory analysis (`zad1.py`)

Explores the raw Iris measurements and species labels:
- summarizes feature distributions with descriptive statistics,
- compares species via histograms and boxplots,
- visualizes relationships between pairs of features with regression lines and correlation.

Representative figures:

![Sepal length histogram](plots/zad1_hist_sepal_length.png)
![Petal length boxplot](plots/zad1_box_petal_length.png)
![Regression: sepal length vs sepal width](plots/zad1_scatter_sepal_length_vs_sepal_width.png)

### 2. Clustering (`zad2.py`)

Discovers structure in the data without using labels:
- normalizes features and fits KMeans for several values of \(k\),
- uses the elbow method to choose a sensible number of clusters,
- inspects how clusters separate in different 2D projections of the feature space.

Representative figures:

![Elbow method – WCSS vs k](plots/zad2_wcss_vs_k.png)
![Clusters in feature space](plots/zad2_clusters_grid.png)

### 3. Classification (`zad3.py`)

Trains and evaluates a **k‑nearest neighbors (k‑NN)** classifier:
- normalizes features and scans several values of \(k\),
- measures test accuracy for all features and for selected feature pairs,
- builds confusion matrices for the best-performing settings.

Representative figures:

![Accuracy vs k – all features](plots/zad3_accuracy_all_features.png)
![Accuracy vs k – petal length & petal width](plots/zad3_accuracy_petal_length_petal_width.png)

### 4. Feature engineering and pipelines (`zad4.py`)

Adds non-linear interactions and wraps the workflow into reusable pipelines:
- baseline pipeline: scaling + logistic regression,
- extended pipeline: scaling + polynomial features + logistic regression,
- compares training accuracy and estimates which original features contribute most.

Representative figures:

![Aggregated feature importance](plots/zad4_feature_importance.png)
![Baseline vs engineered pipeline](plots/zad4_pipeline_performance.png)

### 5. Model comparison (`zad5.py`)

Compares several standard classifiers on the same train/test split:
- k‑NN,
- logistic regression,
- random forest.

Each model is trained on the same data and evaluated on the same test set.

Representative figure:

![Model comparison – test accuracy](plots/zad5_model_accuracy_comparison.png)

### 6. Cross-validation and tuning (`zad6.py`)

Studies how k‑NN performance depends on its main hyperparameter \(k\):
- performs stratified 5‑fold cross-validation,
- sweeps \(k\) from 1 to 20,
- records mean CV accuracy to identify a stable region of good performance.

Representative figure:

![k-NN CV accuracy vs k](plots/zad6_knn_cv_accuracy_vs_k.png)

### 7. Evaluation metrics (`zad7.py`)

Looks beyond overall accuracy to understand error patterns:
- computes a full classification report for k‑NN on the test set,
- shows F1‑scores per class,
- visualizes the confusion matrix with counts per cell.

Representative figures:

![Per-class F1-score](plots/zad7_f1_per_class.png)
![Confusion matrix](plots/zad7_confusion_matrix.png)

### 8. Imbalanced datasets (`zad8.py`)

Simulates class imbalance and shows how it affects model behaviour:
- builds an imbalanced training set by undersampling some classes,
- trains identical k‑NN models on balanced vs imbalanced data,
- compares per‑class accuracy on a common test set.

Representative figure:

![Balanced vs imbalanced per-class accuracy](plots/zad8_class_accuracy_balanced_vs_imbalanced.png)

### 9. Model interpretability (`zad9.py`)

Uses a random forest as a simple, global interpretability tool:
- fits a random forest on the normalized features,
- extracts feature importance scores,
- highlights which measurements the model relies on most.

Representative figure:

![Random forest feature importance](plots/zad9_feature_importance_random_forest.png)

### 10. End-to-end ML pipeline (`zad10.py`)

Combines the previous ideas into a compact end‑to‑end workflow:
- builds a pipeline with scaling and distance‑weighted k‑NN,
- splits the data into training and hold‑out validation sets,
- runs stratified cross-validation and evaluates the final model on unseen data.

Representative figures:

![Cross-validation accuracy per fold](plots/zad10_cv_fold_accuracy.png)
![End-to-end pipeline performance](plots/zad10_pipeline_performance.png)

### Overview

- **`zad1.py`**: understand the data and feature distributions.
- **`zad2.py`**: discover natural groupings via KMeans.
- **`zad3.py`**: train and evaluate a k‑NN classifier.
- **`zad4.py`**: add engineered features and organize steps into pipelines.
- **`zad5.py`**: compare multiple model families on the same task.
- **`zad6.py`**: tune hyperparameters with cross-validation.
- **`zad7.py`**: inspect detailed metrics and error patterns.
- **`zad8.py`**: study how imbalance affects model behaviour.
- **`zad9.py`**: gain insight into feature importance.
- **`zad10.py`**: run a small, realistic end‑to‑end ML pipeline.

Together, these scripts form a compact pipeline that goes from **data exploration**, through **unsupervised learning**, to **supervised classification**, with all main results documented as diagrams in the `plots/` directory.

