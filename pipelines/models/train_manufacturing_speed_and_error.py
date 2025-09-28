# pip install scikit-learn pandas joblib
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, precision_recall_fscore_support
)

# ====== YOUR CSV HEADERS (exact) ======
TS_COL = "Timestamp"
SPEED_TARGET = "Production_Speed_units_per_hr"
ERR_COL      = "Error_Rate_%"

NUM_CANDIDATES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Predictive_Maintenance_Score",
    "Quality_Control_Defect_Rate_%",  # helps both targets
]
CAT_CANDIDATES = [
    "Machine_ID",
    "Operation_Mode",
    "Efficiency_Status",
]
# ======================================

def time_split(df: pd.DataFrame, ts_col: str, train_frac=0.8, val_frac=0.1):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col)
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train+n_val]
    test  = df.iloc[n_train+n_val:]
    return train, val, test

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df[TS_COL], errors="coerce")
    df["hour"]  = dt.dt.hour
    df["dow"]   = dt.dt.dayofweek
    df["month"] = dt.dt.month
    return df

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def choose_existing(df: pd.DataFrame, wanted):
    return [c for c in wanted if c in df.columns]

def build_speed_regressor(num_cols, cat_cols):
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
        ("rf", RandomForestRegressor(n_estimators=500, random_state=7, n_jobs=-1))
    ])

def build_high_error_classifier(num_cols, cat_cols):
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
    ap.add_argument("--train_csv", default="data/manufacturing.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--pred_csv", help="optional CSV to score")
    ap.add_argument("--pred_out", default="out/mf_preds.csv")
    ap.add_argument("--high_err_thresh", type=float, default=2.0)
    args = ap.parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    if TS_COL not in df.columns:
        print(f"[ERROR] '{TS_COL}' not found. Columns: {list(df.columns)}"); sys.exit(1)

    df = add_time_features(df)

    num_cols = choose_existing(df, NUM_CANDIDATES)
    cat_cols = choose_existing(df, CAT_CANDIDATES)
    if not num_cols:
        print("[ERROR] None of the expected numeric columns were found."); sys.exit(1)

    # ---------- SPEED REGRESSION ----------
    if SPEED_TARGET not in df.columns:
        print(f"[WARN] '{SPEED_TARGET}' not in CSV — skipping speed model.")
        speed_model = None
    else:
        df_speed = df.dropna(subset=[SPEED_TARGET]).copy()
        df_speed = coerce_numeric(df_speed, num_cols + [SPEED_TARGET])
        features = num_cols + cat_cols + ["hour","dow","month"]
        tr, va, te = time_split(df_speed, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr[SPEED_TARGET].astype(float)
        X_va, y_va = va[features], va[SPEED_TARGET].astype(float)
        X_te, y_te = te[features], te[SPEED_TARGET].astype(float)

        speed_model = build_speed_regressor(num_cols, cat_cols)
        speed_model.fit(X_tr, y_tr)

        mae_va = mean_absolute_error(y_va, speed_model.predict(X_va))
        r2_va  = r2_score(y_va, speed_model.predict(X_va))
        mae_te = mean_absolute_error(y_te, speed_model.predict(X_te))
        r2_te  = r2_score(y_te, speed_model.predict(X_te))
        print(f"[MF-Speed:VAL ] MAE={mae_va:.3f} R2={r2_va:.3f}")
        print(f"[MF-Speed:TEST] MAE={mae_te:.3f} R2={r2_te:.3f}")

        # refit on train+val before saving
        speed_model.fit(pd.concat([X_tr,X_va]), pd.concat([y_tr,y_va]))
        speed_path = str(Path(args.models_dir) / "mf_speed.joblib")
        joblib.dump({"pipeline": speed_model, "target": SPEED_TARGET,
                     "num_cols": num_cols, "cat_cols": cat_cols}, speed_path)
        print(f"[MF-Speed] Saved -> {speed_path}")

        # model card
        card = {
            "target": SPEED_TARGET,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {
                "val":  {"MAE": float(mae_va), "R2": float(r2_va)},
                "test": {"MAE": float(mae_te), "R2": float(r2_te)}
            },
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour","dow","month"]}
        }
        with open(Path(args.models_dir) / "mf_speed_modelcard.json", "w") as f:
            json.dump(card, f, indent=2)

    # ---------- HIGH-ERROR CLASSIFIER ----------
    if ERR_COL not in df.columns:
        print(f"[WARN] '{ERR_COL}' not in CSV — skipping error model.")
        err_model = None
    else:
        df_err = df.dropna(subset=[ERR_COL]).copy()
        df_err = coerce_numeric(df_err, num_cols + [ERR_COL])
        df_err["high_error"] = (df_err[ERR_COL].astype(float) >= args.high_err_thresh).astype(int)

        features = num_cols + cat_cols + ["hour","dow","month"]
        tr, va, te = time_split(df_err, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr["high_error"]
        X_va, y_va = va[features], va["high_error"]
        X_te, y_te = te[features], te["high_error"]

        err_model = build_high_error_classifier(num_cols, cat_cols)
        err_model.fit(X_tr, y_tr)

        def eval_split(tag, X, y):
            p  = err_model.predict_proba(X)[:,1]
            yh = (p >= 0.5).astype(int)
            auc = roc_auc_score(y, p)
            pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary")
            print(f"[MF-Error:{tag}] AUC={auc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
            return {"AUC": float(auc), "Precision": float(pr), "Recall": float(rc), "F1": float(f1)}

        m_val  = eval_split("VAL",  X_va, y_va)
        m_test = eval_split("TEST", X_te, y_te)

        # refit on train+val before saving
        err_model.fit(pd.concat([X_tr,X_va]), pd.concat([y_tr,y_va]))
        err_path = str(Path(args.models_dir) / "mf_high_error.joblib")
        joblib.dump({"pipeline": err_model, "label": "high_error",
                     "threshold": args.high_err_thresh,
                     "num_cols": num_cols, "cat_cols": cat_cols}, err_path)
        print(f"[MF-Error] Saved -> {err_path}")

        # model card
        card = {
            "label": "high_error",
            "threshold": args.high_err_thresh,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {"val": m_val, "test": m_test},
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour","dow","month"]}
        }
        with open(Path(args.models_dir) / "mf_error_modelcard.json", "w") as f:
            json.dump(card, f, indent=2)

    # ---------- Optional batch predictions ----------
    if args.pred_csv and (speed_model is not None or err_model is not None):
        dfp = pd.read_csv(args.pred_csv)
        dfp = add_time_features(dfp)

        # align columns
        for c in num_cols:
            if c not in dfp.columns: dfp[c] = np.nan
        for c in cat_cols:
            if c not in dfp.columns: dfp[c] = ""

        dfp = coerce_numeric(dfp, num_cols + ([ERR_COL] if ERR_COL in dfp.columns else []))

        feats = num_cols + cat_cols + ["hour","dow","month"]
        Xp = dfp[feats]

        if speed_model is not None:
            dfp["speed_pred"] = np.round(speed_model.predict(Xp), 2)

        if err_model is not None:
            pe = err_model.predict_proba(Xp)[:,1]
            dfp["high_error_prob"] = np.round(pe, 3)
            dfp["high_error_pred"] = (pe >= 0.5).astype(int)

        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(args.pred_out, index=False)
        print(f"[MF] Wrote predictions -> {args.pred_out}")

if __name__ == "__main__":
    main()
