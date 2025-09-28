
import argparse, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# ==== EDIT THESE IF YOUR COLUMNS DIFFER ====
TARGET = "estimated_time_hours"   # numeric ETA in hours
NUM_COLS = [
    "traffic_congestion_index",
    "port_congestion_delay_hours",
    "customs_processing_delay_hours",
    "loading_unloading_time",
    "handling_equipment_availability",
    "weather_condition_severity",
    "lead_time_days",
    "historical_demand",
    "fuel_consumption_rate",
    "driver_behavior_score",
    "route_risk_level",
    "delay_minutes"            # optional if you have it (helps)
]
CAT_COLS = ["route_id", "vehicle_id", "Route_Key"]  # Route_Key if you created it during ingestion
# ===========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/logistics.csv")
    ap.add_argument("--model_out", default="pipelines/models/logistics_eta.joblib")
    ap.add_argument("--pred_csv", help="optional CSV to predict")
    ap.add_argument("--pred_out", default="out/logistics_eta_pred.csv")
    args = ap.parse_args()

    Path("out").mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.train_csv)
    df = df.dropna(subset=[TARGET])  # need target for training

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    model = Pipeline(steps=[
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=7, n_jobs=-1
        ))
    ])

    model.fit(X, y)
    pred = model.predict(X)
    print(f"[ETA] MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.3f}")

    joblib.dump({"pipeline": model, "target": TARGET, "num_cols": NUM_COLS, "cat_cols": CAT_COLS}, args.model_out)
    print(f"[ETA] Saved model -> {args.model_out}")

    if args.pred_csv:
        dfp = pd.read_csv(args.pred_csv)
        Xp = dfp.reindex(columns=NUM_COLS + CAT_COLS, fill_value=0)
        yhat = model.predict(Xp)
        dfp["eta_pred_hours"] = yhat.round(2)
        dfp.to_csv(args.pred_out, index=False)
        print(f"[ETA] Wrote predictions -> {args.pred_out}")

if __name__ == "__main__":
    main()
