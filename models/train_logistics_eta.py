
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


try:
    from ml_report_generator import generate_ml_report_html
except ImportError:
    try:
        from models.ml_report_generator import generate_ml_report_html
    except ImportError:
        print("[WARN] ml_report_generator not found, HTML report will be skipped")
        def generate_ml_report_html(*args, **kwargs):
            pass

# Config / defaults
TARGET_REG = "delivery_time_deviation"
TS_COL = "timestamp"
DELAY_THRESHOLD = None  
RANDOM_SEED = 42


def _read_csv(path: str) -> pd.DataFrame:
    """Read CSV with error handling"""
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[ERROR] File not found: {path}")
        print(f"[INFO] Current working directory: {Path.cwd()}")
        sys.exit(1)
    return pd.read_csv(path)


def _basic_clean_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning and feature engineering"""
    df = df.copy()
    # Parse timestamp if present
    if TS_COL in df.columns:
        try:
            df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
        except Exception:
            df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    else:
        df[TS_COL] = pd.NaT

    # Time features
    df["hour"] = df[TS_COL].dt.hour.fillna(0).astype(int)
    df["dow"] = df[TS_COL].dt.dayofweek.fillna(0).astype(int)
    df["month"] = df[TS_COL].dt.month.fillna(0).astype(int)

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Fill string NaNs
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in object_cols:
        df[c] = df[c].astype(str).fillna("NA")

    return df


def _select_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Choose a reasonable default list of numeric and categorical features"""
    prefer_numeric = [
        "fuel_consumption_rate",
        "eta_variation_hours",
        "traffic_congestion_level",
        "warehouse_inventory_level",
        "loading_unloading_time",
        "handling_equipment_availability",
        "shipping_costs",
        "lead_time_days",
        "iot_temperature",
        "driver_behavior_score",
        "fatigue_monitoring_score",
        "delay_probability",
        "port_congestion_level",
    ]
    prefer_categorical = [
        "order_fulfillment_status",
        "weather_condition_severity",
        "supplier_reliability_score",
        "route_risk_level",
        "risk_classification",
        "cargo_condition_status",
        "customs_clearance_time",
        "Route_Key",
        "vehicle_id",
        "Day",
    ]

    exclude = [TARGET_REG, "hour", "dow", "month", TS_COL]
    numeric = [c for c in prefer_numeric if c in df.columns and c != TARGET_REG]
    categorical = [c for c in prefer_categorical if c in df.columns]
    
    
    # Fallback: any other numeric columns not target
    extra_num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in numeric + [TARGET_REG]]
    for c in extra_num[:5]:  # Limit to 5 extra features to avoid overfitting
        if c not in numeric:
            numeric.append(c)
    
    # Limit categories to reasonable amount
    categorical = [c for c in categorical if df[c].nunique() < 500]
    
    print(f"[INFO] Selected {len(numeric)} numeric features: {numeric}")
    print(f"[INFO] Selected {len(categorical)} categorical features: {categorical}")
    
    return numeric, categorical


def time_split(df: pd.DataFrame, ts_col: str, train_frac=0.8, val_frac=0.1):
    """Time-based split"""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    print(f"[INFO] Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_eta_regressor(num_cols, cat_cols):
    """Build pipeline for ETA regression"""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols + ["hour", "dow", "month"]),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1))
    ])


def build_delay_classifier(num_cols, cat_cols):
    """Build pipeline for delay classification"""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols + ["hour", "dow", "month"]),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return Pipeline([
        ("pre", pre),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED))
    ])


def main():
    ap = argparse.ArgumentParser(description="Train logistics ETA + delay classifier")
    ap.add_argument("--train_csv", required=True, help="Path to cleaned logistics CSV")
    ap.add_argument("--models_dir", default="models", help="Directory to save models and cards")
    ap.add_argument("--pred_csv", help="optional CSV to score")
    ap.add_argument("--pred_out", default="out/logistics_preds.csv")
    ap.add_argument("--delay_threshold", type=float, default=None, help="Delay threshold (auto if None)")
    args = ap.parse_args()

    print(f"[INFO] Current working directory: {Path.cwd()}")
    print(f"[INFO] Training CSV: {args.train_csv}")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)

    df = _read_csv(args.train_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.train_csv}")
    print(f"[INFO] Columns: {list(df.columns)[:10]}...")
    
    if TS_COL not in df.columns:
        print(f"[WARN] '{TS_COL}' not found. Available columns: {list(df.columns)}")
        # Try to find alternative timestamp column
        ts_candidates = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if ts_candidates:
            TS_COL_LOCAL = ts_candidates[0]
            print(f"[INFO] Using alternative timestamp column: {TS_COL_LOCAL}")
            df[TS_COL] = df[TS_COL_LOCAL]
        else:
            print(f"[WARN] No timestamp column found, creating synthetic one")
            df[TS_COL] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

    df = _basic_clean_and_features(df)
    num_cols, cat_cols = _select_features(df)

    if not num_cols:
        print("[ERROR] No numeric columns found.")
        sys.exit(1)

    #ETA REGRESSION
    eta_model = None
    eta_card = {}
    
    if TARGET_REG not in df.columns:
        print(f"[WARN] '{TARGET_REG}' not in CSV — skipping ETA model.")
        print(f"[INFO] Available columns: {list(df.columns)}")
    else:
        print(f"[INFO] Training ETA regression model...")
        df_eta = df.dropna(subset=[TARGET_REG]).copy()
        df_eta[TARGET_REG] = pd.to_numeric(df_eta[TARGET_REG], errors="coerce")
        df_eta = df_eta.dropna(subset=[TARGET_REG])
        
        print(f"[INFO] After cleaning: {len(df_eta)} rows with valid {TARGET_REG}")
        
        features = num_cols + cat_cols + ["hour", "dow", "month"]
        tr, va, te = time_split(df_eta, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr[TARGET_REG].astype(float)
        X_va, y_va = va[features], va[TARGET_REG].astype(float)
        X_te, y_te = te[features], te[TARGET_REG].astype(float)

        eta_model = build_eta_regressor(num_cols, cat_cols)
        print("[INFO] Fitting ETA model...")
        eta_model.fit(X_tr, y_tr)

        mae_va = mean_absolute_error(y_va, eta_model.predict(X_va))
        r2_va = r2_score(y_va, eta_model.predict(X_va))
        mae_te = mean_absolute_error(y_te, eta_model.predict(X_te))
        r2_te = r2_score(y_te, eta_model.predict(X_te))
        
        print(f"[Logistics-ETA:VAL ] MAE={mae_va:.3f} R2={r2_va:.3f}")
        print(f"[Logistics-ETA:TEST] MAE={mae_te:.3f} R2={r2_te:.3f}")

        # Refit on train+val before saving
        eta_model.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
        eta_path = str(Path(args.models_dir) / "logistics_eta.joblib")
        joblib.dump({"pipeline": eta_model, "target": TARGET_REG,
                     "num_cols": num_cols, "cat_cols": cat_cols}, eta_path)
        print(f"[Logistics-ETA] Saved -> {eta_path}")

        # Model card
        eta_card = {
            "target": TARGET_REG,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {
                "val": {"MAE": float(mae_va), "R2": float(r2_va)},
                "test": {"MAE": float(mae_te), "R2": float(r2_te)}
            },
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour", "dow", "month"]}
        }
        with open(Path(args.models_dir) / "logistics_eta_modelcard.json", "w") as f:
            json.dump(eta_card, f, indent=2)

    #  DELAY CLASSIFIER
    global DELAY_THRESHOLD
    if args.delay_threshold is not None:
        DELAY_THRESHOLD = args.delay_threshold
    elif DELAY_THRESHOLD is None and TARGET_REG in df.columns:
        df_for_thresh = df.dropna(subset=[TARGET_REG])
        if len(df_for_thresh) > 0:
            DELAY_THRESHOLD = float(df_for_thresh[TARGET_REG].mean() + df_for_thresh[TARGET_REG].std())
            print(f"[INFO] Auto-calculated delay threshold: {DELAY_THRESHOLD:.3f}")
        else:
            DELAY_THRESHOLD = 1.0

    delay_model = None
    delay_card = {}
    
    if TARGET_REG not in df.columns:
        print(f"[WARN] '{TARGET_REG}' not in CSV — skipping delay classifier.")
    else:
        print(f"[INFO] Training delay classifier...")
        df_delay = df.dropna(subset=[TARGET_REG]).copy()
        df_delay[TARGET_REG] = pd.to_numeric(df_delay[TARGET_REG], errors="coerce")
        df_delay = df_delay.dropna(subset=[TARGET_REG])
        df_delay["high_delay"] = (df_delay[TARGET_REG].astype(float) >= DELAY_THRESHOLD).astype(int)
        
        print(f"[INFO] High delay cases: {df_delay['high_delay'].sum()} / {len(df_delay)} ({df_delay['high_delay'].mean()*100:.1f}%)")

        features = num_cols + cat_cols + ["hour", "dow", "month"]
        tr, va, te = time_split(df_delay, TS_COL, 0.8, 0.1)

        X_tr, y_tr = tr[features], tr["high_delay"]
        X_va, y_va = va[features], va["high_delay"]
        X_te, y_te = te[features], te["high_delay"]

        delay_model = build_delay_classifier(num_cols, cat_cols)
        print("[INFO] Fitting delay classifier...")
        delay_model.fit(X_tr, y_tr)

        def eval_split(tag, X, y):
            p = delay_model.predict_proba(X)[:, 1]
            yh = (p >= 0.5).astype(int)
            auc = roc_auc_score(y, p) if y.sum() > 0 else 0.0
            pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
            print(f"[Logistics-Delay:{tag}] AUC={auc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
            return {"AUC": float(auc), "Precision": float(pr), "Recall": float(rc), "F1": float(f1)}

        m_val = eval_split("VAL", X_va, y_va)
        m_test = eval_split("TEST", X_te, y_te)

        # Refit on train+val before saving
        delay_model.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
        delay_path = str(Path(args.models_dir) / "logistics_delay_clf.joblib")
        joblib.dump({"pipeline": delay_model, "label": "high_delay",
                     "threshold": DELAY_THRESHOLD,
                     "num_cols": num_cols, "cat_cols": cat_cols}, delay_path)
        print(f"[Logistics-Delay] Saved -> {delay_path}")

        # Model card
        delay_card = {
            "label": "high_delay",
            "threshold": float(DELAY_THRESHOLD),
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "metrics": {"val": m_val, "test": m_test},
            "features": {"numeric": num_cols, "categorical": cat_cols, "time": ["hour", "dow", "month"]}
        }
        with open(Path(args.models_dir) / "logistics_delay_modelcard.json", "w") as f:
            json.dump(delay_card, f, indent=2)

    #Optional batch predictions
    if args.pred_csv and (eta_model is not None or delay_model is not None):
        print(f"[INFO] Generating predictions for {args.pred_csv}...")
        dfp = pd.read_csv(args.pred_csv)
        dfp = _basic_clean_and_features(dfp)

        # Align columns
        for c in num_cols:
            if c not in dfp.columns:
                dfp[c] = np.nan
        for c in cat_cols:
            if c not in dfp.columns:
                dfp[c] = ""

        # Coerce numeric
        for c in num_cols + [TARGET_REG]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        feats = num_cols + cat_cols + ["hour", "dow", "month"]
        Xp = dfp[feats]

        if eta_model is not None:
            dfp["eta_pred"] = np.round(eta_model.predict(Xp), 2)
            print(f"[INFO] Added eta_pred column")

        if delay_model is not None:
            pe = delay_model.predict_proba(Xp)[:, 1]
            dfp["delay_prob"] = np.round(pe, 3)
            dfp["delay_pred"] = (pe >= 0.5).astype(int)
            print(f"[INFO] Added delay_prob and delay_pred columns")

        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(args.pred_out, index=False)
        print(f"[Logistics] Wrote predictions -> {args.pred_out}")

    # Generate summary matching manufacturing format
    summary = {
        "models": [],
        "cards": {}
    }
    if eta_model is not None:
        summary["models"].append("logistics_eta.joblib")
        summary["cards"]["eta"] = eta_card
    if delay_model is not None:
        summary["models"].append("logistics_delay_clf.joblib")
        summary["cards"]["delay"] = delay_card
    
    # Generate HTML report
    try:
        pred_sample = []
        if args.pred_out and Path(args.pred_out).exists():
            pred_df = pd.read_csv(args.pred_out)
            pred_sample = pred_df.head(20).to_dict('records')

        report_path = Path("out") / "logistics_ml_report.html"
        generate_ml_report_html(
            domain="logistics",
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


if __name__ == "__main__":
    main()