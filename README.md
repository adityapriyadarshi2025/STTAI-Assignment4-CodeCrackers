# Assignment 4: UrbanNest Rent Predictor

**Hugging Face Demo:** https://huggingface.co/spaces/Priyadarshi101/urbannest-rent-predictor

**Team:** CodeCrackers

## Overview

A rent prediction pipeline for UrbanNest Analytics covering Mumbai, Delhi, Pune, and Hisar. Compares three hyperparameter optimization strategies on a Random Forest regressor, tracks experiments with trackio, serves predictions via a Streamlit frontend, containerized with Docker, deployed on Hugging Face Spaces.

## Best Model

| | Value |
|---|---|
| **Method** | Grid Search |
| **n_estimators** | 200 |
| **max_depth** | 25 |
| **min_samples_split** | 2 |
| **Test MAE** | ₹12,420.35 |

### Note on target leakage
The raw dataset included a `Price_per_sqft` column (equal to `price / Size_ft²`) that leaks the target. We dropped it during preprocessing since a real user cannot provide rent-per-sqft without already knowing the rent. The reported Test MAE reflects the honest predictive performance after removing the leak.

## Optimizer Comparison (60 trials, 5-fold CV)

| Method | Best CV MAE | Best Params |
|---|---|---|
| **Grid Search** 🏆 | ₹13,276.83 | `n_estimators=200, max_depth=25, min_samples_split=2` |
| Random Search | ₹13,307.13 | `n_estimators=142, max_depth=24, min_samples_split=2` |
| Bayesian (Optuna) | ₹13,495.13 | `n_estimators=155, max_depth=24, min_samples_split=2` |

All three methods converged to effectively the same region of the hyperparameter space, with a total spread of only ~₹218 (1.6%) across methods — within cross-validation noise. Grid Search won by a narrow margin because `{200, 25, 2}` landed exactly on a grid point.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

```bash
docker build -t urbannest-rent .
docker run -p 8501:8501 urbannest-rent
```

Visit `http://localhost:8501`.

