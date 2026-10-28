# Term Deposit Prediction Service

A small machine learning service that predicts whether a bank client will subscribe to a term deposit after a marketing call. The model is served as a web API with FastAPI and packaged with Docker.

## Data

UCI Bank Marketing dataset: https://archive.ics.uci.edu/dataset/222/bank

1. Download and unzip it.
2. Put `bank-full.csv` in a folder called `data/`.

The `duration` column (call length) is dropped, because it is only known after the call ends and would leak the outcome.

## Model

A scikit-learn pipeline (one-hot encoding + scaling + logistic regression), saved as `model.joblib`.
Test ROC AUC: about 0.77.

## Run it

### 1. Train the model

```
uv sync
uv run python train.py
```

This creates `model.joblib`.

### 2. Run with Docker

```
docker build -t term-deposit .
docker run -p 9696:9696 term-deposit
```

### 3. Try it

Open http://localhost:9696/docs, click `POST /predict`, then "Try it out", and send:

```json
{
  "age": 35, "job": "management", "marital": "married",
  "education": "tertiary", "default": "no", "balance": 1500,
  "housing": "yes", "loan": "no", "contact": "cellular",
  "day": 15, "month": "may", "campaign": 2, "pdays": -1,
  "previous": 0, "poutcome": "unknown"
}
```

The response contains the probability of subscribing. A health check is available at `/health`.

## Project files

| File | Purpose |
|---|---|
| `train.py` | Trains the model and saves `model.joblib` |
| `predict.py` | FastAPI service |
| `Dockerfile` | Builds the container image |
| `pyproject.toml`, `uv.lock` | Exact dependency versions |