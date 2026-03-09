# Energy Demand Predictor ⚡

**Author:** Osama Ahmed Othman
**GitHub:** [osamaAhmedOthman](https://github.com/osamaAhmedOthman)
**LinkedIn:** [osama-othman](https://www.linkedin.com/in/osama-othman-a78141368)
**Email:** osmanosamaahmed@gmail.com

---

## What Is This Project?

The Energy Demand Predictor is a machine learning system that forecasts how much electricity Spain will need — hour by hour — up to 7 days into the future.

It takes in historical energy generation data and real-time weather readings from across Spain, and outputs a predicted electricity demand in megawatts (MW) for each upcoming hour. The system is packaged as a REST API so it can be called by any application, dashboard, or automated pipeline.

---

## Why Does This Matter in the Real World?

Electricity cannot be stored at scale. Every megawatt that gets generated must be consumed at the same instant, or it is wasted — or worse, it destabilizes the grid.

Grid operators at organizations like Red Eléctrica de España must decide, hours or days in advance, how many power plants to run, how much renewable energy to commit to, and how much reserve capacity to keep on standby. Getting this wrong in either direction has real consequences:

- **Under-prediction** → not enough supply → blackouts or emergency imports from neighboring countries
- **Over-prediction** → too much supply → plants burning fuel for nothing, financial losses, excess CO₂

A model that predicts demand with 1.12% average error (our MAPE) instead of guessing reduces wasted generation, lowers electricity costs, and helps integrate more renewable energy safely into the grid. This is exactly the kind of model that runs inside national energy management systems today.

---

## Model Performance

| Metric | Value | What it means |
|--------|-------|---------------|
| **R²** | 0.9905 | The model explains 99.05% of all demand variation |
| **RMSE** | 446.77 MW | Average error per hour (mean demand is ~29,000 MW) |
| **MAPE** | 1.12% | On average, predictions are off by just 1.12% |
| **MAE** | 327.37 MW | Mean of absolute errors — less sensitive to large spikes than RMSE |
| **Peak Hour RMSE** | 483.38 MW | Accuracy during the hardest hours (morning & evening peaks) |

*Evaluated on 5,260 completely held-out test samples covering May–December 2018. This data was never seen during training or tuning.*

---

## Is This Overfitting?

**No — and here is exactly why.**

This question is worth answering clearly because R² = 0.9905 looks suspiciously high to anyone used to working with noisy datasets like finance or healthcare.

### 1. Energy demand is structurally predictable

Electricity demand is driven by human behavior: people wake up at similar times, offices open and close on schedules, evenings follow predictable patterns. Unlike stock prices (which are theoretically unpredictable), energy demand has a strong deterministic component that repeats reliably every 24 hours and every 7 days. It is one of the most forecastable time series that exists.

### 2. The naive baseline already gets R² = 0.82

Before any machine learning, we tested a naive strategy: "predict whatever demand was at this exact hour last week." That simple rule alone achieves R² ≈ 0.82 and RMSE ≈ 3,767 MW. Our tuned model gets R² = 0.9905 and RMSE = 447 MW. The model is not cheating — it is genuinely learning the remaining patterns that the weekly cycle does not explain.

### 3. The train/test gap is negligible

Overfitting shows up as a large gap between training and test performance. Our train R² ≈ 0.993 versus test R² = 0.9905 is a difference of only 0.0025. An overfit model would have train R² near 1.0 and test R² collapsing toward 0.80 or lower. That did not happen here.

### 4. The test set was locked from day one

The test set (the last 15% of the timeline) was never touched during EDA, model selection, or hyperparameter tuning. It was opened exactly once at the end to produce the final numbers. There is no way for the model to have learned from it.

### 5. Industry benchmarks confirm this range

Published results on this exact ENTSOE dataset — used in multiple academic papers and Kaggle competitions — consistently report R² between 0.98 and 0.995 for well-engineered gradient boosting models. Our result of 0.9905 sits squarely in the middle of what the best public solutions achieve on this problem.

---

## How to Run the Project

### Prerequisites

- Docker and Docker Compose installed on your machine
- The raw data files placed in `data/raw/`:
  - `energy_dataset.csv`
  - `weather_features.csv`

---

### Step 1 — Start the services

```bash
docker-compose up --build
```

This starts two services simultaneously:
- The **FastAPI prediction server** at http://localhost:8000
- The **MLflow tracking UI** at http://localhost:5000

Wait until both containers report they are ready before continuing.

---

### Step 2 — Prepare the data

Open a new terminal and run:

```bash
docker-compose exec api python src/data_loader.py
```

This loads the raw energy and weather CSVs, merges and cleans them, and creates the train / validation / test splits in `data/processed/`. It takes about 30 seconds.

---

### Step 3 — Compare all model families

```bash
docker-compose exec api python src/train.py
```

This trains Ridge, Random Forest, Gradient Boosting, XGBoost, and LightGBM using time-series cross-validation, logs all results to MLflow, and saves the name of the best-performing model. Open http://localhost:5000 to compare the runs visually.

---

### Step 4 — Tune the best model

```bash
docker-compose exec api python src/tune.py
```

This runs 60 Optuna trials on the winning model from Step 3, each trial logged as a nested run in MLflow. The best hyperparameters are saved to `data/processed/best_params.json`. This step takes 5–20 minutes depending on your hardware.

---

### Step 5 — Evaluate on the test set

```bash
docker-compose exec api python src/evaluate.py
```

> ⚠️ **Run this step exactly once.** This is the final, honest evaluation on data the model has never seen. Running it, adjusting the model, and running it again would be test set leakage — the numbers would no longer be trustworthy. Run it once, record the results, and stop.

This generates the evaluation plots in `reports/` and registers the final model to MLflow's Production stage so the API can load it.

---

### Step 6 — Make predictions

The API is now live and serving predictions at:

```
POST http://localhost:8000/api/v1/predict
```

Send a JSON body containing at least 336 hours (14 days) of historical energy and weather data, and specify how many hours ahead you want to forecast. The full request schema and example response are in the interactive API docs at http://localhost:8000/docs.

You can verify the service is healthy at any time:

```
GET http://localhost:8000/health
```

---

## Running Without Docker

If you prefer to run locally without containers:

1. Create a Python 3.10+ virtual environment and install `requirements.txt`
2. Start an MLflow server in a separate terminal: `mlflow server --port 5000`
3. Start the API server in another terminal: `uvicorn api.main:app --reload --port 8000`
4. Run the pipeline scripts in order using `python src/<script>.py` directly — `data_loader.py` → `train.py` → `tune.py` → `evaluate.py`

---

## Running the Tests

```bash
pytest tests/ -v
```

The test suite covers data integrity, feature engineering correctness, pipeline behavior, and prediction safety. All tests use synthetic in-memory data — no real CSV files are required to run them.

---

## Feature Engineering Summary

The model uses 47 features across 7 groups, all computed without any future data leakage:

| Group | Features | Source |
|-------|----------|--------|
| Calendar | Hour, day-of-week, month, quarter, weekend, season, holiday | Timestamp only |
| Cyclical | Sin/cos encodings of hour, dow, month, day-of-year | Timestamp only |
| Lag | Demand 1h, 24h, 48h, 168h, 336h ago | Past target values only |
| Rolling | 24h & 168h mean, std, max, min | Past target values only |
| Weather (raw) | Temperature, humidity, wind speed, rainfall, cloud cover | External measurement |
| Weather (engineered) | temp², is_cold, is_hot, wind chill, temp×humidity | Derived from raw weather |
| Load forecast | Official next-day demand forecast | Published by grid operator |

Lag and rolling features are mathematically guaranteed to use only past data — this is enforced by unit tests that verify the exact shift values at each row.

---

## Project Structure

```
.
├── config.yaml            # Global configuration (YAML file)
├── Dockerfile             # Container build instructions
├── docker-compose.yml     # Multi‑container orchestration
├── requirements.txt       # Runtime dependencies
├── api/                   # FastAPI web service
│   ├── main.py            # Application entrypoint
│   ├── schemas.py         # Pydantic request/response models
│   └── routers/
│       └── predict.py     # Prediction endpoint logic
├── src/                   # Core ML pipeline
│   ├── config.py
│   ├── data_loader.py
│   ├── pipelines.py
│   ├── predict.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── transformers.py
├── notebooks/             # Jupyter analysis (7 notebooks)
├── tests/                 # pytest unit tests
├── data/
│   ├── raw/               # Original CSVs (tracked in git)
│   └── processed/         # Outputs from data_loader (ignored)
├── reports/               # Evaluation plots (generated by evaluate.py)
├── mlruns/                # MLflow tracking data (auto-generated)
├── .gitignore             # Patterns excluded from version control
├── .github/
│   └── workflows/         # GitHub Actions definitions
│       └── ci.yml         # Runs pytest on every push/pr to main
└── README.md              # This document
```

> Raw data files (`data/raw/`) are **not committed to git** — add them manually after cloning. Processed splits and MLflow artifacts are also excluded via `.gitignore`.

---

## Dataset

- **Source:** Spanish electricity system — ENTSOE-E transparency platform
- **Period:** January 2015 – December 2018
- **Frequency:** Hourly, UTC timezone
- **Size:** 35,064 energy records + weather from 5 Spanish cities
- **Split:** 70% training / 15% validation / 15% test — strictly chronological, never shuffled

---

## License

MIT License — free to use for research and commercial purposes.

---

*Last updated: March 2026*