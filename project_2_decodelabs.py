# =============================================================
# DecodeLabs AI Internship — Project 2
# Data Classification Using AI: KNN on the Iris Dataset
# Batch: 2026 | Powered by DecodeLabs
# =============================================================

# NOTE: Run this once in your terminal if libraries are missing:
# pip install matplotlib seaborn scikit-learn numpy pandas

# ── CELL 1: Import Libraries ──────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ── CELL 2: Load the Dataset ──────────────────────────────────
# Load the built-in Iris dataset from sklearn
iris = load_iris()

# Convert to a DataFrame so we can see it clearly
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Quick look at the data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── CELL 3: Understand the Data ───────────────────────────────
# Check class distribution
print("\nClass distribution:")
print(df['species'].value_counts())

print("\nBasic statistics:")
print(df.describe())

# ── CELL 4: Separate Features and Labels ──────────────────────
# X = input features (what the model learns from)
X = iris.data

# y = target labels (what we want to predict)
y = iris.target

print("\nFeatures shape:", X.shape)
print("Labels shape:", y.shape)

# ── CELL 5: Train-Test Split ──────────────────────────────────
# Split: 80% training, 20% testing
# random_state=42 makes results reproducible
# stratify=y keeps class balance in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ── CELL 6: Feature Scaling ───────────────────────────────────
# StandardScaler makes all features have mean=0 and variance=1
# IMPORTANT: fit ONLY on training data, then transform both sets
# (never fit on test data - that would be data leakage)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
X_test_scaled = scaler.transform(X_test)         # only transform on test

print("\nScaling done.")
print("First training sample (scaled):", X_train_scaled[0])

# ── CELL 7: Train the KNN Model ───────────────────────────────
# KNN with k=5 means: look at 5 nearest neighbors and take majority vote
model = KNeighborsClassifier(n_neighbors=5)

# Fit = the model memorizes the training data
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# ── CELL 8: Make Predictions ──────────────────────────────────
# Predict the class for each test sample
predictions = model.predict(X_test_scaled)

# Map numbers back to names for readability
label_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

print("\nPredicted:", [label_map[p] for p in predictions])
print("Actual:   ", [label_map[a] for a in y_test])

# ── CELL 9: Confusion Matrix ──────────────────────────────────
# Confusion matrix shows exactly where the model got confused
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ── CELL 10: F1 Score and Full Report ────────────────────────
# F1 Score = harmonic mean of Precision and Recall
# 'weighted' accounts for class balance
f1 = f1_score(y_test, predictions, average='weighted')
print(f"\nF1 Score (weighted): {f1:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ── CELL 11: Elbow Method (Finding Optimal K) ─────────────────
# Test different K values to find the best one
error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    error = 1 - f1_score(y_test, preds, average='weighted')
    error_rates.append(error)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, error_rates, marker='o', color='steelblue')
plt.title('Elbow Method: Finding Optimal K')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = list(k_range)[error_rates.index(min(error_rates))]
print(f"Best K found: {best_k}")
