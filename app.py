import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
from config import DATA_CSV, MODEL_DIR, OUT_DIR

st.set_page_config(page_title="Municipal Waste Predictor", page_icon="♻️", layout="wide")
st.title("♻️ Municipal Waste Prediction & Recycling Performance")
st.caption("MongoDB + PyMongo + Machine Learning  |  municipality-level data (not live bin sensors)")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_data():
    """Return (DataFrame, source_text)."""
    try:
        db = config.get_db()
        if not config.USING_MOCK:
            rows = list(db["municipalities"].find({}, {"_id": 0}))
            if rows:
                return pd.DataFrame(rows), f"MongoDB ({db.name}.municipalities)"
    except SystemExit as e:                      # MONGO_URI set but connection failed
        st.warning(str(e))
    except Exception as e:                       # noqa: BLE001
        st.warning(f"MongoDB error: {e}")
    return pd.read_csv(DATA_CSV), "CSV file (MongoDB not connected or empty)"


df, source = load_data()
df = df.dropna(subset=["name", "area"]).copy()
for c in df.columns:
    if c not in ("region", "province", "name", "fee"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["waste_per_capita"] = df["msw"] / df["pop"]
st.sidebar.success(f"Data source: {source}")

tab_over, tab_explore, tab_pred, tab_eval, tab_mongo = st.tabs(
    ["Overview", "Explore", "Predict", "Model evaluation", "MongoDB"])

# ------------------------------------------------------------------ overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municipalities", f"{len(df):,}")
    c2.metric("Total waste (thousand tonnes)", f"{df['msw'].sum() / 1e6:,.0f}")
    c3.metric("Avg waste / person / year", f"{df['waste_per_capita'].mean():.0f} kg")
    c4.metric("Avg separate collection", f"{df['sor'].mean():.1f}%")
    st.subheader("Average separate collection (%) by fee system")
    st.bar_chart(df.groupby("fee")["sor"].mean())
    st.subheader("Average waste per person by region (kg/year)")
    st.bar_chart(df.groupby("region")["waste_per_capita"].mean().sort_values(ascending=False))

# ------------------------------------------------------------------ explore
with tab_explore:
    regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
    colf1, colf2 = st.columns(2)
    reg_sel = colf1.selectbox("Region", regions)
    fee_sel = colf2.selectbox("Fee system", ["All"] + sorted(df["fee"].dropna().unique().tolist()))
    view = df.copy()
    if reg_sel != "All":
        view = view[view["region"] == reg_sel]
    if fee_sel != "All":
        view = view[view["fee"] == fee_sel]
    st.write(f"{len(view):,} municipalities")
    st.dataframe(view[["name", "region", "province", "pop", "fee", "msw", "waste_per_capita", "sor"]].head(300),
                 )
    comp = ["organic", "paper", "glass", "wood", "metal", "plastic", "raee", "texile", "other"]
    st.subheader("Average waste composition (%)")
    st.bar_chart(view[comp].mean())
    st.subheader("Waste per person distribution")
    hist, edges = np.histogram(view["waste_per_capita"].dropna().clip(upper=1500), bins=40)
    st.bar_chart(pd.DataFrame({"municipalities": hist}, index=edges[:-1].round(0)))

# ------------------------------------------------------------------ predict
with tab_pred:
    st.subheader("Predict waste per person and recycling class")
    meta_path = MODEL_DIR / "model_meta.json"
    needed = ["regression_random_forest.joblib", "classifier_decision_tree.joblib",
              "classifier_random_forest.joblib"]
    if not meta_path.exists() or not all((MODEL_DIR / n).exists() for n in needed):
        st.info("Models not found. Run `python run_all.py` first to train and save them.")
    else:
        try:
            meta = json.loads(meta_path.read_text())
            reg_model = joblib.load(MODEL_DIR / needed[0])
            dt_model = joblib.load(MODEL_DIR / needed[1])
            rf_model = joblib.load(MODEL_DIR / needed[2])
        except Exception as e:                   # e.g. different scikit-learn version
            st.error(f"Could not load the models ({e}). Re-run `python run_all.py` on this computer.")
            st.stop()
        rng, dflt = meta["numeric_ranges"], meta["numeric_defaults"]
        with st.form("predict"):
            a, b, c = st.columns(3)
            region = a.selectbox("Region", meta["regions"], index=meta["regions"].index("Lombardia")
                                 if "Lombardia" in meta["regions"] else 0)
            fee = b.selectbox("Fee system", meta["fees"])
            pop = c.number_input("Population", 50, 3_000_000, 5000, step=500)
            a, b, c = st.columns(3)
            area = a.number_input("Area (km²)", 0.5, 1500.0, 30.0)
            alt = b.number_input("Altitude (m)", 0, 2500, int(dflt["alt"]))
            gdp = c.number_input("GDP indicator", 0.0, float(rng["gdp"][1]), float(dflt["gdp"]))
            a, b, c, d = st.columns(4)
            organic = a.slider("Organic %", 0.0, 100.0, float(dflt["organic"]))
            paper = b.slider("Paper %", 0.0, 100.0, float(dflt["paper"]))
            glass = c.slider("Glass %", 0.0, 100.0, float(dflt["glass"]))
            plastic = d.slider("Plastic %", 0.0, 100.0, float(dflt["plastic"]))
            wage = st.number_input("Average wage indicator", 0.0, float(rng["wage"][1]), float(dflt["wage"]))
            go = st.form_submit_button("Predict")
        if go:
            row = {c: dflt.get(c, np.nan) for c in meta["feature_columns"]}
            row.update({"alt": alt, "gdp": gdp, "wage": wage, "organic": organic, "paper": paper,
                        "glass": glass, "plastic": plastic, "region": region, "fee": fee,
                        "d_fee": 1 if fee == "PAYT" else 0,
                        "log_pop": np.log1p(pop), "log_area": np.log1p(area),
                        "log_pden": np.log1p(pop / area), "log_roads": np.log1p(meta["roads_default"])})
            X = pd.DataFrame([row])[meta["feature_columns"]]
            kg = float(reg_model.predict(X)[0])
            m1, m2 = st.columns(2)
            m1.metric("Predicted waste per person", f"{kg:.0f} kg / year")
            m2.metric("Estimated total waste", f"{kg * pop / 1000:,.0f} tonnes / year")
            st.markdown("**Recycling performance class**")
            for name, mdl in [("Decision Tree", dt_model), ("Random Forest", rf_model)]:
                proba = pd.Series(mdl.predict_proba(X)[0], index=mdl.classes_)
                st.write(f"{name}: **{proba.idxmax()}**")
                st.bar_chart(proba.reindex(["Low", "Medium", "High"]))
            st.caption("Decision-support estimate from municipality-level data. Unspecified inputs use dataset "
                       "medians. R² of the regression model is about 0.47, so treat values as rough estimates.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    res_path = OUT_DIR / "results.json"
    if not res_path.exists():
        st.info("Run `python run_all.py` to create the evaluation results.")
    else:
        R = json.loads(res_path.read_text())
        st.subheader("Regression (waste per person)")
        st.dataframe(pd.DataFrame(R["regression"]["results"]).T[["R2", "MAE", "RMSE", "CV_R2_mean"]].round(3))
        st.subheader("Classification (recycling class)")
        st.dataframe(pd.DataFrame(R["classification"]["results"]).T[
            ["Accuracy", "Precision", "Recall", "F1", "CV_Accuracy_mean"]].round(3))
        for img, cap in [("07_confusion_matrices.png", "Confusion matrices"),
                         ("05_regression_actual_vs_predicted.png", "Actual vs predicted"),
                         ("06_regression_feature_importance.png", "Regression feature importance"),
                         ("08_classification_feature_importance.png", "Decision Tree feature importance")]:
            if (OUT_DIR / img).exists():
                st.image(str(OUT_DIR / img), caption=cap)

# ------------------------------------------------------------------ MongoDB
with tab_mongo:
    st.subheader("MongoDB collections")
    if config.USING_MOCK:
        st.info("MongoDB server not connected - set MONGO_URI (e.g. MongoDB Atlas) and run `python run_all.py`.")
    else:
        db = config.get_db()
        names = db.list_collection_names()
        st.write({n: db[n].count_documents({}) for n in names})
        if "municipalities" in names:
            st.markdown("**Aggregation: average separate collection by region**")
            agg = list(db["municipalities"].aggregate([
                {"$group": {"_id": "$region", "avg_sor": {"$avg": "$sor"}, "n": {"$sum": 1}}},
                {"$sort": {"avg_sor": -1}}]))
            st.dataframe(pd.DataFrame(agg).rename(columns={"_id": "region"}).round(2))
        if "model_metrics" in names:
            st.markdown("**Stored model metrics**")
            st.json(list(db["model_metrics"].find({}, {"_id": 0, "created_at": 0})))
        if "predictions" in names:
            st.markdown("**Latest stored predictions**")
            st.dataframe(pd.DataFrame(list(db["predictions"].find({}, {"_id": 0}).limit(50))))
