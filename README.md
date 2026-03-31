# ☀️ Solar Analytics Dashboard

A machine learning-powered Streamlit dashboard for analysing solar energy consumer adoption data.

## Features

| Tab | What it does |
|-----|-------------|
| 📊 Exploration | Distribution plots, correlation heatmap, dataset stats |
| 🤖 Classification | Random Forest predicts `AdoptionLikelihood` with accuracy, F1, confusion matrix |
| 📈 Regression | Predicts `EMI_Willingness` with R² score and residual analysis |
| 🔵 Clustering | K-Means consumer segmentation with elbow curve |
| 🔗 Association Rules | Market basket analysis on Income × Location × Adoption |

## Required Dataset Columns

Your `dataset.csv` must include at minimum:
- `AdoptionLikelihood` — categorical target for classification
- `EMI_Willingness` — numeric/categorical target for regression
- `Income` — used in association rules
- `Location` — used in association rules

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push `app.py`, `dataset.csv`, `requirements.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, set **Main file path** to `app.py`
4. Click Deploy

> **Note:** `dataset.csv` must be in the **same directory** as `app.py` in your repo.
