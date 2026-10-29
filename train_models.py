"""Step 2 - Read data from MongoDB, preprocess, train and evaluate ML models.

Regression     : predict waste per inhabitant (kg/person/year) = msw / pop
                 models -> Linear Regression (baseline), Random Forest Regressor
                 metrics -> R2, MAE, RMSE

Classification : predict recycling performance class (Low / Medium / High)
                 derived from `sor` (% of waste collected separately)
                 models -> Decision Tree, Logistic Regression, Random Forest
                 metrics -> Accuracy, Precision, Recall, F1, confusion matrix

Results (predictions + metrics) are written back to MongoDB.
"""
import json
import warnings

import joblib
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_DIR, OUT_DIR, get_db
from load_data import ensure_loaded

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

CLASS_ORDER = ["Low", "Medium", "High"]
FITTED = {}          # fitted pipelines, saved to models/ at the end
EXTRA = {}           # extra info (grid-search table, per-class reports) saved in results.json
COST_COLS = ["tc", "cres", "csor"]

# Columns that are identifiers, outcomes or direct components of the targets.
# They are NOT used as input features (prevents data leakage).
DROP_COLS = ["name", "province", "istat", "sample",          # identifiers
             "msw", "msw_so", "msw_un", "sor", "wden",       # waste totals; wden = msw/area (leaks the target)
             "tc", "cres", "csor"]                            # costs = outcomes of waste management

LOG_COLS = ["pop", "area", "pden", "roads"]           # right-skewed columns
CAT_COLS = ["region", "fee"]


# ----------------------------------------------------------------------------
# 1. Load from MongoDB + preprocessing
# ----------------------------------------------------------------------------
def load_from_mongo():
    col = ensure_loaded()
    df = pd.DataFrame(list(col.find({}, {"_id": 0})))
    print(f"[mongo] Retrieved {len(df)} documents from municipalities")
    return df


def preprocess(df_raw):
    log = {"rows_raw": len(df_raw)}
    df = df_raw.copy()

    # make numeric columns numeric (None -> NaN)
    for c in df.columns:
        if c not in ("region", "province", "name", "fee"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # fix inconsistent label: "piemonte" -> "Piemonte"
    df["region"] = df["region"].str.strip().replace({"piemonte": "Piemonte"})

    log["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()

    # drop rows without name/area/geometry (6 rows, all geography columns missing)
    df = df.dropna(subset=["name", "area"]).reset_index(drop=True)
    log["rows_after_dropna_name_area"] = len(df)

    # derived targets
    df["waste_per_capita"] = df["msw"] / df["pop"]                   # kg / person / year
    q1, q2 = df["sor"].quantile([1 / 3, 2 / 3])
    df["recycling_class"] = pd.cut(df["sor"], [-np.inf, q1, q2, np.inf],
                                   labels=CLASS_ORDER)
    log["sor_thresholds"] = [round(float(q1), 2), round(float(q2), 2)]

    # log transform skewed numeric columns
    for c in LOG_COLS:
        df[f"log_{c}"] = np.log1p(df[c])
    return df, log


def feature_lists(df):
    drop = DROP_COLS + ["waste_per_capita", "recycling_class"] + LOG_COLS
    X = df.drop(columns=drop)
    num = [c for c in X.columns if c not in CAT_COLS]
    return X, num, CAT_COLS


def make_preprocessor(num, cat, scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def feature_names(pipe):
    return [n.replace("num__", "").replace("cat__", "") for n in pipe.named_steps["prep"].get_feature_names_out()]


# ----------------------------------------------------------------------------
# 2. EDA charts
# ----------------------------------------------------------------------------
def eda_charts(df, log):
    miss = pd.Series(log["missing_before"]).sort_values(ascending=False)
    miss = miss[miss.index.isin(df.columns)]
    fig, ax = plt.subplots(figsize=(8, 5))
    (miss / log["rows_raw"] * 100).plot.barh(ax=ax, color="#4C72B0")
    ax.invert_yaxis(); ax.set_xlabel("% missing values")
    ax.set_title("Missing values per column (raw data)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "01_missing_values.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["waste_per_capita"], bins=50, kde=True, ax=axes[0], color="#55A868")
    axes[0].set_title("Waste per inhabitant (kg/person/year)")
    sns.countplot(x="recycling_class", data=df, order=CLASS_ORDER, ax=axes[1], palette="viridis")
    axes[1].set_title("Recycling class distribution")
    fig.tight_layout(); fig.savefig(OUT_DIR / "02_target_distributions.png", dpi=150); plt.close(fig)

    cols = ["waste_per_capita", "sor", "pop", "pden", "alt", "gdp", "wage", "finance",
            "organic", "paper", "glass", "plastic", "other", "tc"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(OUT_DIR / "03_correlation_heatmap.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(x="fee", y="sor", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Separate collection (%) by fee system")
    order = df.groupby("region")["waste_per_capita"].median().sort_values().index
    sns.boxplot(y="region", x="waste_per_capita", data=df, order=order, ax=axes[1],
                color="#8172B2", fliersize=1)
    axes[1].set_title("Waste per inhabitant by region")
    fig.tight_layout(); fig.savefig(OUT_DIR / "04_fee_and_region.png", dpi=150); plt.close(fig)

    # population distribution (truncated at the 99th percentile) + waste vs population
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    cap = df["pop"].quantile(0.99)
    sns.histplot(df.loc[df["pop"] <= cap, "pop"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Population (truncated at 99th percentile)"); axes[0].set_xlabel("Inhabitants")
    axes[1].scatter(df["pop"], df["msw"] / 1e6, s=6, alpha=.4, color="#55A868")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Population (log scale)"); axes[1].set_ylabel("Municipal waste, thousand tonnes (log scale)")
    axes[1].set_title("Total waste vs population")
    fig.tight_layout(); fig.savefig(OUT_DIR / "10_population_and_waste.png", dpi=150); plt.close(fig)

    # mean waste composition
    comp = ["organic", "paper", "glass", "wood", "metal", "plastic", "raee", "texile", "other"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    df[comp].mean().sort_values().plot.barh(ax=ax, color="#8172B2")
    ax.set_xlabel("Mean share of waste (%)"); ax.set_title("Average waste composition (available values)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "11_waste_composition.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------
def run_regression(df):
    X, num, cat = feature_lists(df)
    y = df["waste_per_capita"]
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE)
    Xtr, Xte, ytr, yte = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]

    models = {
        "Linear Regression": Pipeline([("prep", make_preprocessor(num, cat, True)),
                                       ("model", LinearRegression())]),
        "Random Forest Regressor": Pipeline([("prep", make_preprocessor(num, cat, False)),
                                             ("model", RandomForestRegressor(
                                                 n_estimators=300, min_samples_leaf=2,
                                                 n_jobs=-1, random_state=RANDOM_STATE))]),
    }
    results, preds = {}, {}
    for name, pipe in models.items():
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2")
        pipe.fit(Xtr, ytr)
        FITTED[name] = pipe
        p = pipe.predict(Xte)
        preds[name] = p
        results[name] = {
            "R2": r2_score(yte, p),
            "MAE": mean_absolute_error(yte, p),
            "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
            "CV_R2_mean": float(cv.mean()), "CV_R2_std": float(cv.std()),
            "R2_train": r2_score(ytr, pipe.predict(Xtr)),
        }
        print(f"[reg] {name:25s} R2={results[name]['R2']:.3f}  "
              f"MAE={results[name]['MAE']:.1f}  RMSE={results[name]['RMSE']:.1f}  "
              f"CV-R2={cv.mean():.3f}+-{cv.std():.3f}")

    # charts for the Random Forest
    rf = models["Random Forest Regressor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, models):
        ax.scatter(yte, preds[name], s=8, alpha=.5)
        lim = [yte.min(), yte.max()]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Actual (kg/person/year)"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR2={results[name]['R2']:.3f}, MAE={results[name]['MAE']:.1f}")
    fig.tight_layout(); fig.savefig(OUT_DIR / "05_regression_actual_vs_predicted.png", dpi=150); plt.close(fig)

    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=feature_names(rf))
    imp = imp.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot.barh(ax=ax, color="#C44E52")
    ax.set_title("Random Forest Regressor - top 15 feature importances")
    fig.tight_layout(); fig.savefig(OUT_DIR / "06_regression_feature_importance.png", dpi=150); plt.close(fig)

    top_features = {k.replace("num__", "").replace("cat__", ""): float(v) for k, v in imp.head(10).items()}
    return results, preds, idx_test, yte, top_features, len(idx_train), len(idx_test)


def run_classification(df):
    X, num, cat = feature_lists(df)
    y = df["recycling_class"].astype(str)
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE,
                                           stratify=y)
    Xtr, Xte, ytr, yte = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]

    # Decision Tree: tune depth / leaf size with 5-fold CV (training data only)
    dt_pipe = Pipeline([("prep", make_preprocessor(num, cat, False)),
                        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    grid = GridSearchCV(dt_pipe, {"model__max_depth": [3, 5, 7, 10, None],
                                  "model__min_samples_leaf": [1, 5, 10, 20]},
                        cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(Xtr, ytr)
    best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    gr = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").head(6)
    EXTRA["grid_top"] = [{"max_depth": ("None" if pd.isna(r["param_model__max_depth"]) else int(r["param_model__max_depth"])),
                          "min_samples_leaf": int(r["param_model__min_samples_leaf"]),
                          "cv_f1_macro": float(r["mean_test_score"]), "std": float(r["std_test_score"])}
                         for _, r in gr.iterrows()]
    EXTRA["grid_size"] = int(len(grid.cv_results_["params"]))
    print(f"[clf] Decision Tree best params: {best_params}")

    models = {
        "Decision Tree": grid.best_estimator_,
        "Logistic Regression": Pipeline([("prep", make_preprocessor(num, cat, True)),
                                         ("model", LogisticRegression(max_iter=2000))]),
        "Random Forest Classifier": Pipeline([("prep", make_preprocessor(num, cat, False)),
                                              ("model", RandomForestClassifier(
                                                  n_estimators=300, min_samples_leaf=2,
                                                  n_jobs=-1, random_state=RANDOM_STATE))]),
    }
    results, preds, cms, reports = {}, {}, {}, {}
    for name, pipe in models.items():
        if name != "Decision Tree":
            pipe.fit(Xtr, ytr)
        FITTED[name] = pipe
        p = pipe.predict(Xte)
        preds[name] = p
        cms[name] = confusion_matrix(yte, p, labels=CLASS_ORDER)
        reports[name] = classification_report(yte, p, labels=CLASS_ORDER, output_dict=True)
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="accuracy")
        results[name] = {
            "Accuracy": accuracy_score(yte, p),
            "Precision": precision_score(yte, p, average="macro"),
            "Recall": recall_score(yte, p, average="macro"),
            "F1": f1_score(yte, p, average="macro"),
            "CV_Accuracy_mean": float(cv.mean()), "CV_Accuracy_std": float(cv.std()),
            "Accuracy_train": accuracy_score(ytr, pipe.predict(Xtr)),
        }
        r = results[name]
        print(f"[clf] {name:25s} Acc={r['Accuracy']:.3f}  P={r['Precision']:.3f}  "
              f"R={r['Recall']:.3f}  F1={r['F1']:.3f}")

    # confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, name in zip(axes, models):
        sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
        ax.set_title(f"{name}\nAccuracy={results[name]['Accuracy']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUT_DIR / "07_confusion_matrices.png", dpi=150); plt.close(fig)

    dt = models["Decision Tree"]
    imp = pd.Series(dt.named_steps["model"].feature_importances_, index=feature_names(dt))
    imp = imp.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot.barh(ax=ax, color="#DD8452")
    ax.set_title("Decision Tree - top 12 feature importances")
    fig.tight_layout(); fig.savefig(OUT_DIR / "08_classification_feature_importance.png", dpi=150); plt.close(fig)
    top_features = {k.replace("num__", "").replace("cat__", ""): float(v) for k, v in imp.head(10).items()}

    # model comparison chart
    comp = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comp.plot.bar(ax=ax, rot=0, colormap="viridis")
    ax.set_ylim(0, 1); ax.set_title("Classification model comparison (test set, macro average)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT_DIR / "09_classification_comparison.png", dpi=150); plt.close(fig)

    return (results, preds, idx_test, yte, cms, reports, top_features, best_params,
            len(idx_train), len(idx_test))


def run_sensitivity(df):
    """Same Random Forest regression, but ALSO using the cost columns (tc, cres, csor) as inputs.
    Costs are outcomes of waste management, so this is reported only as a sensitivity check."""
    drop = [c for c in DROP_COLS if c not in COST_COLS] + ["waste_per_capita", "recycling_class"] + LOG_COLS
    X = df.drop(columns=drop)
    num = [c for c in X.columns if c not in CAT_COLS]
    y = df["waste_per_capita"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", make_preprocessor(num, CAT_COLS, False)),
                     ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                     n_jobs=-1, random_state=RANDOM_STATE))]).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    out = {"R2": r2_score(yte, p), "MAE": mean_absolute_error(yte, p)}
    print(f"[sens] RF regression WITH cost columns: R2={out['R2']:.3f}  MAE={out['MAE']:.1f}")
    return out


def save_models(df):
    """Save fitted pipelines + metadata (used by the Streamlit app)."""
    X, num, cat = feature_lists(df)
    meta = {
        "feature_columns": list(X.columns),
        "numeric_defaults": {c: float(df[c].median()) for c in X.columns if c in num and not c.startswith("log_")},
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())]
                           for c in ["pop", "area", "alt", "gdp", "wage", "organic", "paper", "glass", "plastic"]},
        "roads_default": float(df["roads"].median()),
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "fees": sorted(df["fee"].dropna().unique().tolist()),
    }
    files = {"Random Forest Regressor": "regression_random_forest.joblib",
             "Decision Tree": "classifier_decision_tree.joblib",
             "Random Forest Classifier": "classifier_random_forest.joblib"}
    for name, fn in files.items():
        joblib.dump(FITTED[name], MODEL_DIR / fn, compress=3)
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[models] Saved {len(files)} models to {MODEL_DIR}")


# ----------------------------------------------------------------------------
# 4. Save results to MongoDB + disk
# ----------------------------------------------------------------------------
def save_results(df, reg, clf, prep_log, sens):
    db = get_db()
    r_res, r_pred, r_idx, r_y, r_top, r_ntr, r_nte = reg
    (c_res, c_pred, c_idx, c_y, cms, reports, c_top, dt_params, c_ntr, c_nte) = clf
    now = datetime.now(timezone.utc)

    # predictions collection: one document per test municipality
    pred_col = db["predictions"]
    pred_col.drop()
    docs = []
    reg_best = "Random Forest Regressor"
    for i, idx in enumerate(r_idx):
        docs.append({"istat": int(df.loc[idx, "istat"]), "name": df.loc[idx, "name"],
                     "region": df.loc[idx, "region"], "task": "regression",
                     "actual_waste_per_capita": float(r_y.loc[idx]),
                     "predicted_waste_per_capita": float(r_pred[reg_best][i]),
                     "created_at": now})
    for i, idx in enumerate(c_idx):
        docs.append({"istat": int(df.loc[idx, "istat"]), "name": df.loc[idx, "name"],
                     "region": df.loc[idx, "region"], "task": "classification",
                     "actual_class": str(c_y.loc[idx]),
                     "predicted_class": str(c_pred["Decision Tree"][i]),
                     "created_at": now})
    pred_col.insert_many(docs)
    pred_col.create_index([("task", 1), ("istat", 1)])

    # model_metrics collection
    metrics_col = db["model_metrics"]
    metrics_col.drop()
    metrics_docs = []
    for name, m in r_res.items():
        metrics_docs.append({"task": "regression", "model": name, "metrics": m,
                             "train_size": r_ntr, "test_size": r_nte, "created_at": now})
    for name, m in c_res.items():
        metrics_docs.append({"task": "classification", "model": name, "metrics": m,
                             "confusion_matrix": cms[name].tolist(), "labels": CLASS_ORDER,
                             "train_size": c_ntr, "test_size": c_nte, "created_at": now})
    metrics_col.insert_many(metrics_docs)
    print(f"[mongo] Saved {len(docs)} predictions and {len(metrics_docs)} metric documents")

    summary = {
        "preprocessing": prep_log,
        "regression": {"target": "waste_per_capita = msw / pop (kg/person/year)",
                       "train_size": r_ntr, "test_size": r_nte,
                       "results": r_res, "top_features": r_top,
                       "sensitivity_with_cost_columns": sens},
        "classification": {"target": "recycling_class from sor (Low/Medium/High)",
                           "train_size": c_ntr, "test_size": c_nte,
                           "decision_tree_params": {k: (None if v is None else v) for k, v in dt_params.items()},
                           "results": c_res,
                           "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
                           "per_class_report_decision_tree": reports["Decision Tree"],
                           "per_class_reports_all": reports,
                           "decision_tree_grid_top": EXTRA.get("grid_top"),
                           "decision_tree_grid_size": EXTRA.get("grid_size"),
                           "top_features_decision_tree": c_top},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = []
    for n, m in r_res.items():
        rows.append({"task": "regression", "model": n, **m})
    for n, m in c_res.items():
        rows.append({"task": "classification", "model": n, **m})
    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    return summary


def main():
    df_raw = load_from_mongo()
    df, prep_log = preprocess(df_raw)
    print(f"[prep] {prep_log['rows_raw']} raw rows -> {len(df)} after cleaning; "
          f"sor thresholds {prep_log['sor_thresholds']}")
    eda_charts(df, prep_log)
    reg = run_regression(df)
    clf = run_classification(df)
    sens = run_sensitivity(df)
    save_models(df)
    return save_results(df, reg, clf, prep_log, sens)


if __name__ == "__main__":
    main()
