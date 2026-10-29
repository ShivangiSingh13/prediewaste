# Municipal Waste Prediction & Recycling Performance - MongoDB + PyMongo + Machine Learning

Dataset: `data/waste_fee.csv` (4,341 Italian municipalities, 39 columns).
This dataset is municipality-level data, not live bin-sensor data, so the project does not claim bin-overflow prediction.

## What the project does
| Task | Target | Models | Metrics |
|---|---|---|---|
| Regression | waste per inhabitant = msw / pop (kg/person/year) | Linear Regression, Random Forest Regressor | R2, MAE, RMSE |
| Classification | recycling class (Low / Medium / High) from `sor` | Decision Tree, Logistic Regression, Random Forest | Accuracy, Precision, Recall, F1, confusion matrix |

MongoDB (`waste_db`): `municipalities` (source rows), `predictions` (test-set predictions), `model_metrics` (evaluation results).
A Streamlit dashboard shows the data, makes predictions with the saved models, and displays metrics and MongoDB contents.

## Files
- `config.py`           - MongoDB connection (PyMongo), reads `MONGO_URI` / `MONGO_DB` from environment or `.env`
- `load_data.py`        - CSV -> MongoDB, creates indexes
- `train_models.py`     - reads from MongoDB, preprocesses, trains/evaluates, saves predictions + metrics to MongoDB, charts, models
- `mongo_operations.py` - CRUD / query / aggregation / index demo (log in `outputs/mongodb_operations_log.txt`)
- `extra_analysis.py`   - statistical test, regional table, regression error analysis, descriptive statistics (used in the report)
- `make_diagrams.py`    - workflow + architecture diagrams
- `app.py`              - Streamlit dashboard
- `run_all.py`          - runs everything in one command
- `build_report.js`     - (optional) regenerates `Project_Report.docx` from `outputs/results.json`
- `outputs/`, `models/` - charts, metrics (`results.json`, `metrics_summary.csv`), saved models

## Setup with MongoDB Atlas
1. Create a free M0 cluster at https://cloud.mongodb.com
2. Database Access -> add a user (avoid special characters in the password).
3. Network Access -> Add IP Address -> allow your IP (or 0.0.0.0/0 for a demo).
4. Connect -> Drivers -> Python -> copy the connection string.
5. Copy `.env.example` to `.env` and paste the string into `MONGO_URI`
   (or set it in the terminal: `export MONGO_URI="mongodb+srv://..."` / `set MONGO_URI=...` on Windows).

## Run
```
pip install -r requirements.txt
python run_all.py          # load MongoDB -> train/evaluate -> MongoDB demo -> diagrams
streamlit run app.py       # dashboard
```
Individual steps: `python load_data.py`, `python train_models.py`, `python mongo_operations.py`.
If `MONGO_URI` is set but the connection fails, the program stops with an error. If nothing is configured and no local
MongoDB runs, an in-memory `mongomock` database is used so the code still runs (data is not saved).

## Data leakage handling
- `msw`, `msw_so`, `msw_un`, `sor` are excluded from the features (the targets are built from them).
- `wden` is excluded: it equals `msw / area`, i.e. a direct function of the regression target.
- Cost columns (`tc`, `cres`, `csor`) are excluded from the main models because they are outcomes of waste volume;
  a sensitivity run with them is reported in the report (Section 11.9).

## Notes
- Metrics in `outputs/` come from running this code on the supplied CSV (random_state = 42).
- The shipped `models/*.joblib` files were saved with the scikit-learn version used here; if loading fails on another
  version, just run `python run_all.py` again to recreate them.
- Do not submit your real `.env` file or database credentials.
