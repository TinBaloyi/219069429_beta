# pip install scikit-learn pandas joblib
import argparse, pandas as pd, joblib, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, precision_recall_fscore_support
from pathlib import Path

# ==== EDIT THESE TO MATCH YOUR DATA ====
SPEED_TARGET = "Production_Speed_units_per_hr"
ERROR_PCT     = "Error_Rate_%"            # numeric %
HIGH_ERR_THRESH = 2.0                     # tweak if needed

NUM_COLS = [
    "cycle_time",
    "queue_len",
    "downtime_flag",          # if bool, we’ll treat it as numeric 0/1
    "utilization_pct",        # include if you have it
    "shift_hours",            # include if you have it
]
CAT_COLS = ["machine_id", "line_id"]      # include any other categorical IDs
# ======================================

def train_speed(df):
    df = df.dropna(subset=[SPEED_TARGET])
    y = df[SPEED_TARGET].astype(float)
    X = df[NUM_COLS + CAT_COLS].copy()
    if "downtime_flag" in X:
        X["downtime_flag"] = X["downtime_flag"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    model = Pipeline(steps=[
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=7, n_jobs=-1))
    ])
    model.fit(X, y)
    pred = model.predict(X)
    print(f"[MF-Speed] MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.3f}")
    return model

def train_high_error(df):
    df = df.dropna(subset=[ERROR_PCT]).copy()
    df["high_error"] = (df[ERROR_PCT].astype(float) >= HIGH_ERR_THRESH).astype(int)
    y = df["high_error"]
    X = df[NUM_COLS + CAT_COLS].copy()
    if "downtime_flag" in X:
        X["downtime_flag"] = X["downtime_flag"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    clf = Pipeline(steps=[
        ("pre", pre),
        ("gb", GradientBoostingClassifier(random_state=7))
    ])
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p>=0.5).astype(int), average="binary")
    print(f"[MF-Error] AUC={auc:.3f}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")
    return clf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/manufacturing.csv")
    ap.add_argument("--speed_model_out", default="pipelines/models/mf_speed.joblib")
    ap.add_argument("--error_model_out", default="pipelines/models/mf_high_error.joblib")
    ap.add_argument("--pred_csv", help="optional CSV to predict")
    ap.add_argument("--speed_pred_out", default="out/mf_speed_pred.csv")
    ap.add_argument("--error_pred_out", default="out/mf_high_error_pred.csv")
    args = ap.parse_args()

    Path("out").mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.train_csv)

    speed_model = train_speed(df)
    joblib.dump({"pipeline": speed_model, "target": SPEED_TARGET, "num_cols": NUM_COLS, "cat_cols": CAT_COLS},
                args.speed_model_out)
    print(f"[MF] Saved speed model -> {args.speed_model_out}")

    error_model = train_high_error(df)
    joblib.dump({"pipeline": error_model, "target": "high_error", "threshold": HIGH_ERR_THRESH,
                 "num_cols": NUM_COLS, "cat_cols": CAT_COLS}, args.error_model_out)
    print(f"[MF] Saved high-error model -> {args.error_model_out}")

    if args.pred_csv:
        dfp = pd.read_csv(args.pred_csv)

        # speed predictions
        Xs = dfp.reindex(columns=NUM_COLS + CAT_COLS, fill_value=0)
        if "downtime_flag" in Xs:
            Xs["downtime_flag"] = Xs["downtime_flag"].astype(float)
        dfp["speed_pred"] = speed_model.predict(Xs).round(2)

        # error probability
        pe = error_model.predict_proba(Xs)[:,1]
        dfp["high_error_prob"] = pe.round(3)
        dfp["high_error_pred"] = (pe >= 0.5).astype(int)

        dfp[[*dfp.columns]].to_csv(args.speed_pred_out, index=False)
        dfp[[*dfp.columns]].to_csv(args.error_pred_out, index=False)
        print(f"[MF] Wrote predictions -> {args.speed_pred_out} / {args.error_pred_out}")

if __name__ == "__main__":
    main()

