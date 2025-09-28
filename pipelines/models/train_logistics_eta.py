# pipelines/models/train_logistics_eta.py
# pip install scikit-learn pandas joblib
import argparse, pandas as pd, numpy as np, joblib, sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, precision_recall_fscore_support

TS_COL = "timestamp"

# Full lists based on YOUR header
NUM_CANDIDATES = [
    "fuel_consumption_rate",
    "traffic_congestion_level",
    "warehouse_inventory_level",
    "loading_unloading_time",
    "handling_equipment_availability",
    "weather_condition_severity",
    "port_congestion_level",
    "shipping_costs",
    "supplier_reliability_score",
    "lead_time_days",
    "historical_demand",
    "iot_temperature",
    "customs_clearance_time",
    "driver_behavior_score",
    "fatigue_monitoring_score",
    "disruption_likelihood_score",
    "route_risk_level",
    "vehicle_gps_latitude",
    "vehicle_gps_longitude",
]
CAT_CANDIDATES = [
    "order_fulfillment_status",
    "cargo_condition_status",
    "risk_classification",
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df[TS_COL], errors="coerce")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    return df

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def choose_existing(df: pd.DataFrame, wanted):
    return [c for c in wanted if c in df.columns]

def build_regressor(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols + ["hour","dow","month"]),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=7, n_jobs=-1))
    ])

def build_delay_classifier(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols + ["hour","dow","month"]),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return Pipeline([
        ("pre", pre),
        ("gb", GradientBoostingClassifier(random_state=7))
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/logistics.csv")
    ap.add_argument("--target", default="delivery_time_deviation",
                    choices=["delivery_time_deviation","eta_variation_hours"])
    ap.add_argument("--model_out", default="pipelines/models/logistics_eta.joblib")
    ap.add_argument("--delay_clf_out", default="pipelines/models/logistics_delay_clf.joblib")
    ap.add_argument("--pred_csv", help="optional CSV to predict")
    ap.add_argument("--pred_out", default="out/logistics_eta_pred.csv")
    args = ap.parse_args()

    Path("out").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    # Derive features safely
    df = add_time_features(df)
    num_cols = choose_existing(df, NUM_CANDIDATES)
    cat_cols = choose_existing(df, CAT_CANDIDATES)
    if not num_cols:
        print("[ERROR] None of the expected numeric columns were found. Found:", list(df.columns)); sys.exit(1)

    # Coerce numerics
    df = coerce_numeric(df, num_cols + [args.target, "delay_probability"])

    # --- ETA regression ---
    if args.target not in df.columns:
        print(f"[ERROR] Target '{args.target}' not found. Available columns:", list(df.columns)); sys.exit(1)
    df_eta = df.dropna(subset=[args.target]).copy()
    if df_eta.empty:
        print(f"[ERROR] No rows with a value for '{args.target}'."); sys.exit(1)

    X_eta = df_eta[num_cols + cat_cols + ["hour","dow","month"]]
    y_eta = df_eta[args.target].astype(float)
    reg = build_regressor(num_cols, cat_cols)
    reg.fit(X_eta, y_eta)
    pred_eta = reg.predict(X_eta)
    print(f"[LOG-ETA:{args.target}] MAE={mean_absolute_error(y_eta, pred_eta):.3f}  R2={r2_score(y_eta, pred_eta):.3f}")
    joblib.dump({"pipeline": reg, "target": args.target,
                 "num_cols": num_cols, "cat_cols": cat_cols}, args.model_out)
    print(f"[LOG-ETA] Saved -> {args.model_out}")

    # --- Delay classifier (weak label from delay_probability) ---
    if "delay_probability" in df.columns and df["delay_probability"].notna().any():
        dfc = df.dropna(subset=["delay_probability"]).copy()
        dfc["delayed"] = (dfc["delay_probability"].astype(float) >= 0.5).astype(int)
        Xc = dfc[num_cols + cat_cols + ["hour","dow","month"]]
        yc = dfc["delayed"]
        clf = build_delay_classifier(num_cols, cat_cols)
        clf.fit(Xc, yc)
        pc = clf.predict_proba(Xc)[:,1]
        auc = roc_auc_score(yc, pc)
        p, r, f1, _ = precision_recall_fscore_support(yc, (pc>=0.5).astype(int), average="binary")
        print(f"[LOG-Delay] AUC={auc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        joblib.dump({"pipeline": clf, "label": "delayed",
                     "num_cols": num_cols, "cat_cols": cat_cols}, args.delay_clf_out)
        print(f"[LOG-Delay] Saved -> {args.delay_clf_out}")
    else:
        print("[LOG-Delay] Skipped (no/empty delay_probability).")

    # --- Optional batch predictions ---
    if args.pred_csv:
        dfp = pd.read_csv(args.pred_csv)
        dfp = add_time_features(dfp)
        # Align columns
        for c in num_cols:
            if c not in dfp.columns: dfp[c] = np.nan
        for c in cat_cols:
            if c not in dfp.columns: dfp[c] = ""
        dfp = coerce_numeric(dfp, num_cols)
        Xp = dfp[num_cols + cat_cols + ["hour","dow","month"]]
        yhat = reg.predict(Xp)
        dfp["eta_pred_hours"] = np.round(yhat, 2)

        # delay risk prediction if model exists
        try:
            clf_bundle = joblib.load(args.delay_clf_out)
            clf = clf_bundle["pipeline"]
            pc = clf.predict_proba(Xp)[:,1]
            dfp["delay_risk_pred"] = np.round(pc, 3)
            dfp["delay_flag_pred"] = (pc >= 0.5).astype(int)
        except Exception:
            pass

        dfp.to_csv(args.pred_out, index=False)
        print(f"[LOG-ETA] Wrote predictions -> {args.pred_out}")

if __name__ == "__main__":
    main()
