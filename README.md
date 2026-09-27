# Recommendation Systems — MovieLens Engine

An end-to-end Machine Learning recommendation system implementation developed as part of the Naviotech Solution internship. The project includes both Content-Based Filtering and Collaborative Filtering via Latent Factor Matrix Factorization (TruncatedSVD).

---

## 📌 Project Overview
- **Assigned Category:** Recommendation Systems
- **Dataset:** MovieLens 100K Benchmark Dataset
- **Catalog Size:** 9,742 movies, 100,836 ratings across 610 users
- **Matrix Sparsity:** 98.30%

---

## 🏗️ Architecture & Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed rating distributions across the 0.5 to 5.0 scale.
- Identified the top most-rated films (e.g., *Forrest Gump*, *The Shawshank Redemption*, *Pulp Fiction*).
- Computed the user-item interaction matrix sparsity.

### 2. Model 1: Content-Based Filtering
- Extracted genre metadata signatures using `TfidfVectorizer` (with English stop words removed).
- Computed pairwise **Cosine Similarity** matrices across all items.
- Generated top-N similar titles with calculated similarity confidence scores.

### 3. Model 2: Collaborative Filtering (Matrix Factorization)
- Constructed a high-dimensional User-Item interaction matrix.
- Applied user de-meaning to normalize individual rating bias.
- Reduced dimensions to latent preference components using **TruncatedSVD** ($k=20$).
- Reconstructed predicted ratings to provide personalized recommendations for unrated titles.

---

## 📈 Quantitative Evaluation
Evaluated model reconstruction accuracy against ground-truth user ratings:
- **Root Mean Squared Error (RMSE):** 0.7625
- **Mean Absolute Error (MAE):** 0.5959

---

## 💻 Tech Stack
- **Language:** Python
- **Environment:** Jupyter Notebook
- **Core Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
-
