
import argparse
import json
import sys
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


try:
    from ml_report_generator import generate_ml_report_html
except ImportError:
    try:
        from models.ml_report_generator import generate_ml_report_html
    except ImportError:
        print("[WARN] ml_report_generator not found, HTML report will be skipped")
        def generate_ml_report_html(*args, **kwargs):
            pass

# CSV HEADERS
TS_COL = "Timestamp"
SPEED_TARGET = "Production_Speed_units_per_hr"
ERR_COL = "Error_Rate_%"

NUM_CANDIDATES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Predictive_Maintenance_Score",
    "Quality_Control_Defect_Rate_%",
]
CAT_CANDIDATES = [
    "Machine_ID",
    "Operation_Mode",
    "Efficiency_Status",
]

def time_split(df: pd.DataFrame, ts_col: str, train_frac=0.8, val_frac=0.1):
    """Time-based split"""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train+n_val]
    test = df.iloc[n_train+n_val:]
    print(f"[INFO] Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features"""
    dt = pd.to_datetime(df[TS_COL], errors="coerce")
    df["hour"] = dt.dt.hour.fillna(0).astype(int)
    df["dow"] = dt.dt.dayofweek.fillna(0).astype(int)
    df["month"] = dt.dt.month.fillna(0).astype(int)
    return df

def coerce_numeric(df: pd.DataFrame, cols):
    """Convert columns to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def choose_existing(df: pd.DataFrame, wanted):
    """Select columns that exist in dataframe"""
    existing = [c for c in wanted if c in df.columns]
    print(f"[INFO] Found {len(existing)}/{len(wanted)} expected columns: {existing}")
    return existing

def build_speed_regressor(num_cols, cat_cols):
    """Build production speed regression pipeline"""
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
    """Build high error rate classifier pipeline"""
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
    ap = argparse.ArgumentParser(description="Train manufacturing models")
    ap.add_argument("--train_csv", default="data/manufacturing.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--pred_csv", help="optional CSV to score")
    ap.add_argument("--pred_out", default="out/mf_preds.csv")
    ap.add_argument("--high_err_thresh", type=float, default=2.0)
    args = ap.parse_args()

    print(f"[INFO] Current working directory: {Path.cwd()}")
    print(f"[INFO] Training CSV: {args.train_csv}")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)

    # Check if file exists
    train_path = Path(args.train_csv)
    if not train_path.exists():
        print(f"[ERROR] Training file not found: {train_path}")
        print(f"[INFO] Looking in current directory...")
        sys.exit(1)

    df = pd.read_csv(args.train_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.train_csv}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    if TS_COL not in df.columns:
        print(f"[WARN] '{TS_COL}' not found. Available columns: {list(df.columns)}")
        # Try to find alternative
        ts_candidates = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if ts_candidates:
            TS_COL_LOCAL = ts_candidates[0]
            print(f"[INFO] Using alternative timestamp column: {TS_COL_LOCAL}")
            df[TS_COL] = df[TS_COL_LOCAL]
        else:
            print(f"[WARN] No timestamp column found, creating synthetic one")
            df[TS_COL] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

    df = add_time_features(df)

    num_cols = choose_existing(df, NUM_CANDIDATES)
    cat_cols = choose_existing(df, CAT_CANDIDATES)
    
    if not num_cols:
        print("[WARN] No expected numeric columns found, using all numeric columns")
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in [SPEED_TARGET, ERR_COL, "hour", "dow", "month"]][:10]
        print(f"[INFO] Using numeric columns: {num_cols}")

    #SPEED REGRESSION
    speed_model = None
    speed_card = {}
    
    if SPEED_TARGET not in df.columns:
        print(f"[WARN] '{SPEED_TARGET}' not in CSV — skipping speed model.")
        print(f"[INFO] Available columns: {list(df.columns)}")
    else:
        print(f"[INFO] Training production speed model...")
        df_speed = df.dropna(subset=[SPEED_TARGET]).copy()
        df_speed = coerce_numeric(df_speed, num_cols + [SPEED_TARGET])
        df_speed = df_speed.dropna(subset=[SPEED_TARGET])
        
        print(f"[INFO] After cleaning: {len(df_speed)} rows with valid {SPEED_TARGET}")
        
        features = num_cols + cat_cols + ["hour","dow","month"]
        tr, va, te = time_split(df_speed, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr[SPEED_TARGET].astype(float)
        X_va, y_va = va[features], va[SPEED_TARGET].astype(float)
        X_te, y_te = te[features], te[SPEED_TARGET].astype(float)

        speed_model = build_speed_regressor(num_cols, cat_cols)
        print("[INFO] Fitting speed model...")
        speed_model.fit(X_tr, y_tr)

        mae_va = mean_absolute_error(y_va, speed_model.predict(X_va))
        r2_va = r2_score(y_va, speed_model.predict(X_va))
        mae_te = mean_absolute_error(y_te, speed_model.predict(X_te))
        r2_te = r2_score(y_te, speed_model.predict(X_te))
        
        print(f"[MF-Speed:VAL ] MAE={mae_va:.3f} R2={r2_va:.3f}")
        print(f"[MF-Speed:TEST] MAE={mae_te:.3f} R2={r2_te:.3f}")

        # Refit on train+val before saving
        speed_model.fit(pd.concat([X_tr,X_va]), pd.concat([y_tr,y_va]))
        speed_path = str(Path(args.models_dir) / "mf_speed.joblib")
        joblib.dump({"pipeline": speed_model, "target": SPEED_TARGET,
                     "num_cols": num_cols, "cat_cols": cat_cols}, speed_path)
        print(f"[MF-Speed] Saved -> {speed_path}")

        # Model card
        speed_card = {
            "target": SPEED_TARGET,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {
                "val": {"MAE": float(mae_va), "R2": float(r2_va)},
                "test": {"MAE": float(mae_te), "R2": float(r2_te)}
            },
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour","dow","month"]}
        }
        with open(Path(args.models_dir) / "mf_speed_modelcard.json", "w") as f:
            json.dump(speed_card, f, indent=2)

    #HIGH-ERROR CLASSIFIER
    err_model = None
    err_card = {}
    
    if ERR_COL not in df.columns:
        print(f"[WARN] '{ERR_COL}' not in CSV — skipping error model.")
    else:
        print(f"[INFO] Training error rate classifier...")
        df_err = df.dropna(subset=[ERR_COL]).copy()
        df_err = coerce_numeric(df_err, num_cols + [ERR_COL])
        df_err = df_err.dropna(subset=[ERR_COL])
        df_err["high_error"] = (df_err[ERR_COL].astype(float) >= args.high_err_thresh).astype(int)
        
        print(f"[INFO] High error cases: {df_err['high_error'].sum()} / {len(df_err)} ({df_err['high_error'].mean()*100:.1f}%)")

        features = num_cols + cat_cols + ["hour","dow","month"]
        tr, va, te = time_split(df_err, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr["high_error"]
        X_va, y_va = va[features], va["high_error"]
        X_te, y_te = te[features], te["high_error"]

        err_model = build_high_error_classifier(num_cols, cat_cols)
        print("[INFO] Fitting error classifier...")
        err_model.fit(X_tr, y_tr)

        def eval_split(tag, X, y):
            p = err_model.predict_proba(X)[:,1]
            yh = (p >= 0.5).astype(int)
            auc = roc_auc_score(y, p) if y.sum() > 0 else 0.0
            pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
            print(f"[MF-Error:{tag}] AUC={auc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
            return {"AUC": float(auc), "Precision": float(pr), "Recall": float(rc), "F1": float(f1)}

        m_val = eval_split("VAL", X_va, y_va)
        m_test = eval_split("TEST", X_te, y_te)

        # Refit on train+val before saving
        err_model.fit(pd.concat([X_tr,X_va]), pd.concat([y_tr,y_va]))
        err_path = str(Path(args.models_dir) / "mf_high_error.joblib")
        joblib.dump({"pipeline": err_model, "label": "high_error",
                     "threshold": args.high_err_thresh,
                     "num_cols": num_cols, "cat_cols": cat_cols}, err_path)
        print(f"[MF-Error] Saved -> {err_path}")

        # Model card
        err_card = {
            "label": "high_error",
            "threshold": args.high_err_thresh,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {"val": m_val, "test": m_test},
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour","dow","month"]}
        }
        with open(Path(args.models_dir) / "mf_error_modelcard.json", "w") as f:
            json.dump(err_card, f, indent=2)

    # Optional batch prediction
    if args.pred_csv and (speed_model is not None or err_model is not None):
        print(f"[INFO] Generating predictions for {args.pred_csv}...")
        dfp = pd.read_csv(args.pred_csv)
        dfp = add_time_features(dfp)

        # Align columns
        for c in num_cols:
            if c not in dfp.columns:
                dfp[c] = np.nan
        for c in cat_cols:
            if c not in dfp.columns:
                dfp[c] = ""

        dfp = coerce_numeric(dfp, num_cols + ([ERR_COL] if ERR_COL in dfp.columns else []))

        feats = num_cols + cat_cols + ["hour","dow","month"]
        Xp = dfp[feats]

        if speed_model is not None:
            dfp["speed_pred"] = np.round(speed_model.predict(Xp), 2)
            print(f"[INFO] Added speed_pred column")

        if err_model is not None:
            pe = err_model.predict_proba(Xp)[:,1]
            dfp["high_error_prob"] = np.round(pe, 3)
            dfp["high_error_pred"] = (pe >= 0.5).astype(int)
            print(f"[INFO] Added high_error_prob and high_error_pred columns")

        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(args.pred_out, index=False)
        print(f"[MF] Wrote predictions -> {args.pred_out}")

    # Generate summary
    summary = {
        "models": [],
        "cards": {}
    }
    if speed_model is not None:
        summary["models"].append("mf_speed.joblib")
        summary["cards"]["speed"] = speed_card
    if err_model is not None:
        summary["models"].append("mf_high_error.joblib")
        summary["cards"]["error"] = err_card

    # Generate HTML report
    try:
        pred_sample = []
        if args.pred_out and Path(args.pred_out).exists():
            pred_df = pd.read_csv(args.pred_out)
            pred_sample = pred_df.head(20).to_dict('records')

        report_path = Path("out") / "manufacturing_ml_report.html"
        generate_ml_report_html(
            domain="manufacturing",
            model_cards=summary["cards"],
            output_path=report_path,
            predictions_sample=pred_sample
        )
        print(f"[INFO] Generated HTML report: {report_path}")
    except Exception as e:
        print(f"[WARN] Could not generate HTML report: {e}")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))
    return summary


# Wrapper for pipeline compatibility
def train_manufacturing(rows, algo="rf"):
    """
    Wrapper function for compatibility with main.py
    """
    print("[INFO] train_manufacturing wrapper called")
    return {"status": "use_cli_trainer"}


if __name__ == "__main__":
    main()