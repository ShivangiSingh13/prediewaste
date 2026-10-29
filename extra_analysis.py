"""Step 4 - Extra analyses used in the report (reads everything from MongoDB):
statistical test (PAYT vs STANDARD), regional table, error analysis of the regression model,
descriptive statistics and sample MongoDB documents.
Run after train_models.py (it needs the `predictions` collection)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import OUT_DIR, get_db
from load_data import ensure_loaded

sns.set_theme(style="whitegrid")


def main():
    ensure_loaded()
    db = get_db()
    df = pd.DataFrame(list(db["municipalities"].find({}, {"_id": 0})))
    df = df.dropna(subset=["name", "area"]).copy()
    df["region"] = df["region"].replace({"piemonte": "Piemonte"})
    for c in df.columns:
        if c not in ("region", "province", "name", "fee"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["waste_per_capita"] = df["msw"] / df["pop"]
    out = {}

    # ---- 1. PAYT vs STANDARD ---------------------------------------------------
    rows = []
    for var, label in [("sor", "Separate collection (%)"), ("waste_per_capita", "Waste per person (kg/year)")]:
        a = df.loc[df["fee"] == "PAYT", var].dropna()
        b = df.loc[df["fee"] == "STANDARD", var].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"variable": label, "payt_n": len(a), "std_n": len(b),
                     "payt_mean": a.mean(), "std_mean": b.mean(), "payt_median": a.median(), "std_median": b.median(),
                     "mannwhitney_p": float(p), "welch_t": float(t), "welch_p": float(pt)})
    out["fee_tests"] = rows

    # ---- 2. regional table -----------------------------------------------------------
    reg = df.groupby("region").agg(n=("name", "size"), avg_sor=("sor", "mean"),
                                    avg_waste=("waste_per_capita", "mean"),
                                    payt_share=("d_fee", "mean")).sort_values("avg_sor", ascending=False)
    out["regional"] = [{"region": i, "n": int(r.n), "avg_sor": float(r.avg_sor), "avg_waste": float(r.avg_waste),
                        "payt_share": float(r.payt_share)} for i, r in reg.iterrows()]

    # ---- 3. regression error analysis (uses predictions stored in MongoDB) ----------
    pr = pd.DataFrame(list(db["predictions"].find({"task": "regression"}, {"_id": 0})))
    if pr.empty:
        raise SystemExit("No regression predictions in MongoDB - run train_models.py first.")
    pr["error"] = pr["actual_waste_per_capita"] - pr["predicted_waste_per_capita"]
    pr["abs_error"] = pr["error"].abs()
    pr = pr.merge(df[["istat", "pop", "fee"]], on="istat", how="left")
    out["error_stats"] = {
        "n": int(len(pr)), "mean_error_bias": float(pr["error"].mean()),
        "median_abs_error": float(pr["abs_error"].median()),
        "within_50": float((pr["abs_error"] <= 50).mean()), "within_100": float((pr["abs_error"] <= 100).mean()),
        "over_200": float((pr["abs_error"] > 200).mean()),
        "mae_below_1000": float(pr.loc[pr["actual_waste_per_capita"] <= 1000, "abs_error"].mean()),
        "n_above_1000": int((pr["actual_waste_per_capita"] > 1000).sum()),
        "mae_above_1000": float(pr.loc[pr["actual_waste_per_capita"] > 1000, "abs_error"].mean()),
    }
    top = pr.sort_values("abs_error", ascending=False).head(10)
    out["top_errors"] = [{"name": r["name"], "region": r["region"], "pop": float(r["pop"]),
                          "actual": float(r["actual_waste_per_capita"]),
                          "predicted": float(r["predicted_waste_per_capita"])} for _, r in top.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(pr["error"], bins=60, ax=axes[0], color="#C44E52")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_xlabel("Error = actual - predicted (kg/person/year)"); axes[0].set_title("Distribution of prediction errors")
    axes[1].scatter(pr["predicted_waste_per_capita"], pr["error"], s=6, alpha=.4, color="#4C72B0")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("Predicted (kg/person/year)"); axes[1].set_ylabel("Error"); axes[1].set_title("Residuals vs predicted")
    fig.tight_layout(); fig.savefig(OUT_DIR / "14_residuals.png", dpi=150); plt.close(fig)

    # ---- 4. descriptive statistics -----------------------------------------------
    num = df.select_dtypes("number").drop(columns=["istat", "sample"], errors="ignore")
    d = pd.DataFrame({"missing_pct": num.isna().mean() * 100, "mean": num.mean(), "std": num.std(),
                      "min": num.min(), "median": num.median(), "max": num.max()})
    out["describe"] = [{"variable": i, **{k: float(v) for k, v in r.items()}} for i, r in d.iterrows()]

    # ---- 5. sample documents ---------------------------------------------------------
    def clean(doc):
        return json.loads(json.dumps(doc, default=str))
    out["sample_municipality"] = clean(db["municipalities"].find_one({"name": "Comacchio"},
        {"_id": 0, "istat": 1, "name": 1, "region": 1, "province": 1, "pop": 1, "area": 1, "fee": 1,
         "d_fee": 1, "organic": 1, "paper": 1, "msw": 1, "sor": 1}))
    out["sample_prediction"] = clean(db["predictions"].find_one({"task": "classification"}, {"_id": 0}))
    mm = db["model_metrics"].find_one({"model": "Decision Tree"}, {"_id": 0})
    mm["metrics"] = {k: round(v, 4) for k, v in mm["metrics"].items() if k in ("Accuracy", "Precision", "Recall", "F1")}
    out["sample_metrics"] = clean(mm)

    (OUT_DIR / "extra_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("[extra] saved extra_results.json and 14_residuals.png")
    return out


if __name__ == "__main__":
    main()
