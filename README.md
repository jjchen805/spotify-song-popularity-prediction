# 🎵 Spotify Song Popularity Prediction

An end-to-end machine learning project that predicts whether a song will be **popular**
(top quantile of Spotify popularity) using audio features and metadata.
The project compares multiple modeling strategies and includes an interactive,
Spotify-styled dashboard for real-time inference.

---

## 🚀 Project Highlights
- Full ML lifecycle: data preparation → modeling → evaluation → deployment
- Multiple model families:
  - LASSO Logistic Regression
  - CART & Random Forest
  - XGBoost (with LASSO feature selection and PCA)
- Feature selection using L1 regularization (LASSO)
- Dimensionality reduction using PCA
- Production-style scikit-learn pipelines
- Interactive Dash dashboard with model selection and real-time predictions

---

## 📊 Problem Definition
**Task:** Binary classification  
**Goal:** Predict whether a song is *popular*.

A song is labeled **popular = 1** if its Spotify popularity score falls within the
**top 10%** of the dataset; otherwise **popular = 0**.

**Input Features**
- Audio features: danceability, energy, loudness, tempo, valence, etc.
- Metadata: genre, key, mode, explicit flag, time signature
- Duration (converted to milliseconds)

---

## 🧠 Modeling Strategy

Two parallel modeling pipelines are implemented.

### 1️⃣ LASSO-Based Pipeline

One-Hot Encoding + Standardization
→ L1 Logistic Regression (feature selection)
→ CART / Random Forest / XGBoost

- LASSO performs automatic feature selection by shrinking coefficients to zero
- Selected features are passed to tree-based and boosting models
- Improves interpretability and reduces overfitting

---

### 2️⃣ PCA-Based Pipeline

Ordinal Encoding + Standardization
→ Principal Component Analysis (PCA)
→ CART / Random Forest / XGBoost

- PCA reduces dimensionality while preserving variance
- Enables efficient modeling in lower-dimensional space
- Useful for comparison against sparse, feature-selected models

---

## 🖥️ Interactive Dashboard

The project includes a Spotify-styled **Dash dashboard** that allows users to:
- Adjust audio features and metadata
- Select among trained models:
  - LASSO
  - CART + LASSO
  - Random Forest + LASSO
  - XGBoost + LASSO
  - CART + PCA
  - Random Forest + PCA
  - XGBoost + PCA
- View predicted probability of popularity in real time

All models are loaded as **joblib pipelines**, so preprocessing and inference are
handled automatically.

Run locally:
```bash
python app_advanced.py
```
---
## 🖼 Dashboard Preview
![Dashboard](assets/dashboard.png)

---

## 📈 Example Results

Model	ROC-AUC
LASSO Logistic Regression	~0.68
CART + LASSO	~0.71
Random Forest + LASSO	~0.72
XGBoost + LASSO	~0.74
XGBoost + PCA	~0.72

(Exact results may vary by random seed and train/test split.)

---

## ⚙️ How to Run the Project

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Preprocess data
```bash
python -m src.data_prep \
  --input data/raw/spotify_dataset.csv \
  --output data/processed/spotify_processed.csv
```
3. Train all models
```bash
python -m src.train --data data/processed/spotify_processed.csv
```
4. Evaluate models
```bash
python -m src.evaluate --data data/processed/spotify_processed.csv
```
5. Launch the advanced dashboard
```bash
python app_advanced.py
```
---

👤 Team

Joshua Chen, Robert Chang, Johnathan Wu, Ethan Lam
