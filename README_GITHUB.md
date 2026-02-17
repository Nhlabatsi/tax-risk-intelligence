# 🏛️ Tax Compliance Risk Intelligence Platform

> ML-powered taxpayer risk classification — built for Amdocs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Overview

Classifies taxpayers into **Low / Medium / High** risk using machine learning trained on financial and compliance data — with **zero data leakage**.

## Features
- **Single Prediction** — assess one taxpayer in real time
- **Batch Processing** — upload CSV, get risk for all records
- **Model Insights** — feature importance, overfitting analysis
- **Export** — download assessment reports as CSV

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/tax-risk-intelligence
cd tax-risk-intelligence
pip install -r requirements.txt
# Place risk_model_clean.pkl in root directory
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Select repo → app.py → Deploy
4. Upload risk_model_clean.pkl via Streamlit file uploader

## Project Structure
```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── risk_model_clean.pkl    # Trained model (no leakage)
├── .streamlit/config.toml  # Theme config
└── README.md
```

*Amdocs Tax Compliance · 2025*
