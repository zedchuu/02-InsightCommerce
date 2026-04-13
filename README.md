# 🛍️ InsightCommerce: AI Customer Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-Data%20Engineering-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)

## 📌 Executive Summary
A complete end-to-end Machine Learning pipeline designed to transform raw transactional data into actionable business intelligence. This project utilizes unsupervised clustering to identify customer personas and supervised machine learning to predict real-time customer behavior and churn risk. 

**Live Demo:** Local Dashboard Available

## 🏢 The Business Problem
Marketing departments often rely on blanket campaigns, wasting budget on "Lost" customers while ignoring "At-Risk" VIPs. The goal of this project was to build a data product that automatically segments a customer base using historical purchasing behavior (RFM) and provides a predictive engine for future engagement.

## ⚙️ System Architecture & Methodology

1. **Data Engineering (DuckDB & Pandas):** * Built a vectorized SQL pipeline using DuckDB to process hundreds of thousands of raw invoice rows.
   * Aggregated transactional data into Recency, Frequency, and Monetary (RFM) profiles.

2. **Mathematical Normalization (Numpy & Scikit-Learn):**
   * Applied logarithmic transformations (`np.log1p`) to handle extreme monetary outliers (right-skewed commerce data).
   * Standardized features (`StandardScaler`) to ensure balanced Euclidean distance calculations.

3. **Unsupervised Discovery (K-Means Clustering):**
   * Utilized the Elbow Method and Silhouette Scores to validate the optimal number of customer segments (K=4).
   * Translated mathematical clusters into business logic (Champions, At-Risk Loyalists, Recent Bargain Hunters, The Lost).

4. **Predictive Modeling (Random Forest):**
   * Trained an ensemble classifier to predict a customer's assigned segment based on continuous behavior.
   * **Achieved 97% overall accuracy** on the holdout test set.
   * Extracted Feature Importance (Recency proved to be the #1 driver of customer retention).

## 📊 The Final Data Product (Streamlit)
The pipeline culminates in a `Streamlit` web application designed for non-technical stakeholders. It features:
* A **"What-If" Simulation Engine:** Marketers can dial in a customer's behavior metrics.
* **Real-time AI Inference:** The Random Forest instantly outputs a segmented probability breakdown.
* **3D Dimensionality Visualization:** An interactive Plotly 3D scatter plot dynamically pinpoints the simulated customer within the broader historical database.

## 🚀 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/zedchuu/02-InsightCommerce.git](https://github.com/zedchuu/02-InsightCommerce.git)

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Launch the dashboard directly from the root folder:
   ```bash
   streamlit run streamlit/app.py

*Designed and built by **Adam Zikri Ahmadilfitri** | [Connect on LinkedIn](https://www.linkedin.com/in/adam-zikri-ahmadilfitri-3a0079183)*
