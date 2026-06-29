# 🌸 DecodeLabs AI Internship — Project 2
## Data Classification Using AI: KNN on the Iris Dataset
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch%202026-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
 
---
 
## 📌 Project Overview
 
This is **Project 2** of the DecodeLabs AI Industrial Training Program (Batch 2026).  
The goal is to build a supervised machine learning pipeline that classifies Iris flowers into 3 species using the **K-Nearest Neighbors (KNN)** algorithm.
 
---
 
## 🎯 Objective
 
Build a basic classification model using the Iris dataset by:
- Loading and understanding the dataset
- Applying feature scaling
- Splitting data into training and testing sets
- Training a KNN classifier
- Evaluating performance with a Confusion Matrix and F1 Score
---
 
## 🏗️ Pipeline Architecture (IPO Framework)
 
```
INPUT               PROCESS                 OUTPUT
─────────────────────────────────────────────────────
Iris Dataset   →   Train-Test Split   →   Confusion Matrix
Feature Scaling →  KNN Algorithm      →   F1 Score
```
 
---
 
## 📂 Project Structure
 
```
decodelabs-p2/
│
├── notebooks/
│   └── iris_knn_classifier.ipynb   ← Main Jupyter Notebook
│
├── outputs/
│   ├── confusion_matrix.png        ← Generated after running notebook
│   └── elbow_curve.png             ← Generated after running notebook
│
├── REPORT.md                       ← Project Report
└── README.md                       ← This file
```
 
---
 
## 📊 Dataset: The Iris Benchmark
 
| Property   | Value              |
|------------|--------------------|
| Samples    | 150 (Balanced)     |
| Classes    | 3                  |
| Features   | 4                  |
| Source     | sklearn.datasets   |
 
**Classes:** Setosa · Versicolor · Virginica  
**Features:** Sepal Length · Sepal Width · Petal Length · Petal Width
 
---
 
## ⚙️ Tech Stack
 
- Python 3.11
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
---
 
## 🚀 How to Run
 
**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/decodelabs-p2-iris-knn.git
cd decodelabs-p2-iris-knn
```
 
**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
 
**3. Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/iris_knn_classifier.ipynb
```
 
**4. Run all cells top to bottom**
 
---
 
## 📈 Results
 
| Metric           | Score     |
|------------------|-----------|
| F1 Score         | ~0.97     |
| Best K (Elbow)   | 5         |
| Test Samples     | 30        |
 
> Exact scores will appear after running the notebook.
 
---
 
## 🧠 Key Concepts Demonstrated
 
- **Supervised Learning** — training on labeled data
- **Feature Scaling** — StandardScaler (mean=0, variance=1)
- **Train-Test Split** — 80/20 with stratification
- **KNN Algorithm** — proximity-based majority voting
- **Confusion Matrix** — TP, FP, FN, TN breakdown
- **F1 Score** — harmonic mean of Precision and Recall
- **Elbow Method** — finding optimal K value
---
 
## 👨‍💻 Author
 
**Indrajith Peiris**  
AI Engineer | DecodeLabs Batch 2026  
Sri Lanka 🇱🇰
 
---
 
## 📜 License
 
This project is part of the DecodeLabs Industrial Training Program.  
Built for learning and portfolio purposes.
