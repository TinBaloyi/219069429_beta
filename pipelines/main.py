from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathlib import Path
from datetime import datetime
import os
import sys
import json




BASE_DIR = Path(__file__).resolve().parent           
PROJECT_ROOT = BASE_DIR.parent                       
OUT_DIR = PROJECT_ROOT / "out"


for p in (str(BASE_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# Project modules 

from data_ingestio_pipeline import DataSource, HistoricalData, DataProcessor
from analysis import AnalysisParameters, BottleneckDetector
from bottlneck_profiles import manufacturing_profiles, logistics_profiles
from config import DB_CONFIG


try:
    from pipelines.ml_model import (
        read_csv as ml_read_csv,
        train_manufacturing,
        train_logistics,
        TrainedModelBundle,
        write_json,
    )
except Exception:
    from ml_model import (
        read_csv as ml_read_csv,
        train_manufacturing,
        train_logistics,
        TrainedModelBundle,
        write_json,
    )

# Third-party libs
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.io as pio
import bcrypt


# App setup
app = FastAPI(title="AI Bottleneck Detection & Reporting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_DIR.mkdir(exist_ok=True)
app.mount("/out", StaticFiles(directory=str(OUT_DIR)), name="out")



# DB helpers & models
def get_db_connection():
    """Open a psycopg2 connection using DB_CONFIG (config.py)."""
    return psycopg2.connect(**DB_CONFIG)


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str



# Auth endpoints
@app.post("/auth/login")
def login_user(data: LoginRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, password FROM users WHERE email = %s", (data.email,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user or not bcrypt.checkpw(data.password.encode(), user[3].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {"user": {"id": user[0], "name": user[1], "email": user[2]}}


@app.post("/auth/signup")
def signup_user(data: SignupRequest):
    hashed_pw = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (data.email,))
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Email already exists")

    cur.execute(
        "INSERT INTO users (name, email, password) VALUES (%s, %s, %s) RETURNING id",
        (data.name, data.email, hashed_pw)
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return {"user": {"id": user_id, "name": data.name, "email": data.email}}



# Core pipeline helpers
def _rows_from_csv(file_path: Path, data_type: str) -> list[dict]:
    
    ds = DataSource(
        source_id="src1",
        source_name="LocalCSV",
        data_type=data_type.upper(),
        connection_parameters={},
        authentication_credentials={},
        data_format="csv",
        file_path=str(file_path),
    )
    if not ds.validate_connection():
        raise HTTPException(status_code=400, detail=f"Cannot read file: {file_path}")

    raw_data = ds.import_historical_data()
    hist = HistoricalData("hist1", ds.source_id, "all", raw_data)
    processor = DataProcessor()

    
    cleaned = processor.clean_and_process_data(hist.raw_data, cleaning_columns=None)

    
    try:
        transformed = processor.transform_data(cleaned, transformations=None)
    except Exception:
        transformed = cleaned
    return transformed


def _detect_bottlenecks(rows: list[dict], data_type: str) -> list[dict]:
   
    detector = BottleneckDetector()
    profiles = manufacturing_profiles if data_type.upper() == "MANUFACTURING" else logistics_profiles
    results: list[dict] = []

    for profile in profiles:
        params = AnalysisParameters(
            analysis_id="auto",
            dataset_id=str(profile.metric_column or "na"),
            analysis_type=data_type.upper(),
            group_by_column=profile.group_by_column,
            metric_column=profile.metric_column,
            modelSelection=[profile.group_by_column],
        )
        # Ensure detector knows whether to minimize or maximize
        setattr(params, "mode", getattr(profile, "mode", "min"))

        events = detector.detect_bottlenecks(rows, params)
        for ev in events:
            metric_val = getattr(ev, "metric_value", None)
            severity = float(getattr(ev, "severity_score", 0.0))
            confidence = float(getattr(ev, "confidence_score", 0.0))
            reason = getattr(ev, "reason", "")
            detected_at = getattr(ev, "start_time", datetime.utcnow())

            results.append({
                "bottleneck_type": profile.name,
                "description": getattr(profile, "description", ""),
                "entity": ev.entity_id,
                "metric": profile.metric_column,
                "metric_value": metric_val,
                "reason": reason,
                "detected_at": detected_at.isoformat() if hasattr(detected_at, "isoformat") else str(detected_at),
                "severity_score": severity,
                "confidence_score": confidence,
                "profile_mode": getattr(profile, "mode", "min"),
                "group_by": profile.group_by_column,
                "recommendations": [
                    "Verify staffing/capacity",
                    "Inspect quality gates",
                    "Consider rerouting/retiming",
                ],
            })
    return results


def detect_all_bottlenecks(cleaned_data, profiles, data_type):
    
    all_results = []
    detector = BottleneckDetector()
    for profile in profiles:
        params = AnalysisParameters(
            analysis_id="auto",
            dataset_id="auto",
            analysis_type=data_type,
            group_by_column=profile.group_by_column,
            metric_column=profile.metric_column,
            modelSelection=[profile.group_by_column],
        )
        setattr(params, "mode", profile.mode)
        events = detector.detect_bottlenecks(cleaned_data, params)
        for ev in events:
            all_results.append({
                "bottleneck_type": profile.name,
                "description": profile.description,
                "entity": ev.entity_id,
                "metric": profile.metric_column,
                "metric_value": getattr(ev, "metric_value", None),
                "reason": getattr(ev, "reason", "N/A"),
                "detected_at": getattr(ev, "start_time", datetime.utcnow()),
                "severity_score": float(getattr(ev, "severity_score", 0.0)),
                "confidence_score": float(getattr(ev, "confidence_score", 0.0)),
                "recommendations": [
                    "Verify staffing/capacity",
                    "Inspect quality gates",
                    "Consider rerouting/retiming",
                ],
                "root_cause_analysis": {"profile_mode": profile.mode, "metric": profile.metric_column},
            })
    return all_results


def save_bottleneck_results(results, run_id=None):
    
    if not results:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    for r in results:
        severity_score = float(r.get("severity_score", 0.0))
        confidence_score = float(r.get("confidence_score", 0.0))
        detected_at = r.get("detected_at")
        if isinstance(detected_at, str):
            try:
                detected_at = datetime.fromisoformat(detected_at)
            except Exception:
                detected_at = datetime.utcnow()
        root_cause_analysis = json.dumps(r.get("root_cause_analysis", {"metric": r.get("metric"), "mode": r.get("profile_mode")}))
        recommendations = json.dumps(r.get("recommendations", []))
        cur.execute(
            """
            INSERT INTO bottleneck_results (
                run_id, bottleneck_type, location_identifier,
                severity_score, impact_description, detected_at,
                confidence_score, root_cause_analysis, recommendations
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id or datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                r.get("bottleneck_type", "Unknown"),
                str(r.get("entity", "")),
                severity_score,
                r.get("description", ""),
                detected_at,
                confidence_score,
                root_cause_analysis,
                recommendations,
            ),
        )
    conn.commit()
    cur.close()
    conn.close()


def run_full_bottleneck_pipeline(file_path, data_type, run_id=None, save_results=True):
    
    ds = DataSource("src1", "DynamicSource", data_type, {}, {}, "csv", file_path)
    if not ds.validate_connection():
        return []
    raw_data = ds.import_historical_data()
    hist_data = HistoricalData("hist1", ds.source_id, "all", raw_data)
    processor = DataProcessor()
    profiles = manufacturing_profiles if data_type.upper() == "MANUFACTURING" else logistics_profiles
    cleaning_cols = {p.group_by_column for p in profiles}
    for p in profiles:
        if p.metric_column:
            cleaning_cols.add(p.metric_column)
    cleaned = processor.clean_and_process_data(hist_data.raw_data, cleaning_columns=list(cleaning_cols))
    results = detect_all_bottlenecks(cleaned, profiles, data_type)
    if save_results:
        save_bottleneck_results(results, run_id)
    return results





def _simple_dashboard_html(bottlenecks: list[dict], ml_card: dict, out_dir: str, title="AI Bottleneck Dashboard"):
    os.makedirs(out_dir, exist_ok=True)
    figs = []

    # 1) Top bottlenecks bar
    top = sorted(bottlenecks, key=lambda b: float(b.get("severity_score", 0.0)), reverse=True)[:12]
    if top:
        df_top = pd.DataFrame(
            [{"Entity": r["entity"], "Severity": r.get("severity_score", 0.0), "Type": r["bottleneck_type"]} for r in top]
        )
        figs.append(px.bar(df_top, x="Entity", y="Severity", color="Type", title="Top Bottlenecks (Severity)"))

    # 2) ML “card” – show F1 by domain 
    if ml_card:
        rows = []
        for dom, info in ml_card.items():
            m = info.get("metrics", {})
            rows.append(
                {"Domain": dom, "Accuracy": m.get("accuracy", 0), "Precision": m.get("precision", 0),
                 "Recall": m.get("recall", 0), "F1": m.get("f1", 0)}
            )
        if rows:
            df_ml = pd.DataFrame(rows)
            figs.append(px.bar(df_ml, x="Domain", y="F1", title="Classification F1 by Domain"))

    html_parts = [pio.to_html(fig, include_plotlyjs="cdn", full_html=False) for fig in figs]
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body><h1>{title}</h1>{''.join(html_parts) or '<p>No charts available.</p>'}</body></html>"
    )
    dash_path = os.path.join(out_dir, "dashboard.html")
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html)
    return dash_path



# API Endpoints

@app.get("/")
def root():
    return {
        "status": "API is running!",
        "endpoints": ["/analysis/manufacturing", "/analysis/logistics", "/whatif/{type}", "/dashboard", "/out/*"],
        "csv_expected": str(PROJECT_ROOT),
    }


@app.get("/analysis/{data_type}")
def run_analysis(data_type: str):
    
    file_map = {
        "manufacturing": PROJECT_ROOT / "manufacturing.csv",
        "logistics":     PROJECT_ROOT / "logistics.csv",
    }
    file_path = file_map.get(data_type.lower())
    if not file_path:
        raise HTTPException(status_code=400, detail="Invalid data type (use manufacturing or logistics)")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {file_path}")

    # 1) bottlenecks (also writes to DB)
    results = run_full_bottleneck_pipeline(str(file_path), data_type.upper(), save_results=True)

    
    ml_card = {}
    rows = ml_read_csv(str(file_path))
    if data_type.lower() == "manufacturing":
        bundle, info = train_manufacturing(rows, algo="rf")
        bundle.save(str(OUT_DIR / "manu_model.json"))
        ml_card["manufacturing"] = {"model": "rf", **info}
    else:
        bundle, info = train_logistics(rows, algo="rf")
        bundle.save(str(OUT_DIR / "log_model.json"))
        ml_card["logistics"] = {"model": "rf", **info}
    write_json(str(OUT_DIR / "model_card.json"), ml_card)

    # 3) Reports (HTML)
    b_html = "<!doctype html><html><head><meta charset='utf-8'><title>Bottleneck Report</title><style>body{font-family:system-ui;margin:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f4f4f4}</style></head><body>"
    b_html += "<h1>Bottleneck Report</h1><table><tr><th>Entity</th><th>Type</th><th>Metric</th><th>Value</th><th>Severity</th><th>Reason</th><th>Detected</th></tr>"
    for r in results:
        b_html += (
            f"<tr><td>{r['entity']}</td><td>{r['bottleneck_type']}</td><td>{r.get('metric','')}</td>"
            f"<td>{r.get('metric_value','')}</td><td>{round(r.get('severity_score',0),2)}</td>"
            f"<td>{r.get('reason','')}</td><td>{r.get('detected_at','')}</td></tr>"
        )
    b_html += "</table></body></html>"
    (OUT_DIR / f"{data_type.lower()}_bottlenecks.html").write_text(b_html, encoding="utf-8")

    m_html = f"<!doctype html><html><head><meta charset='utf-8'><title>ML Report</title></head><body><h1>ML Report — {data_type.title()}</h1><pre>{json.dumps(ml_card, indent=2)}</pre></body></html>"
    (OUT_DIR / f"{data_type.lower()}_ml_report.html").write_text(m_html, encoding="utf-8")

    # 4) Offline dashboard
    dash_path = _simple_dashboard_html(results, ml_card, out_dir=str(OUT_DIR), title=f"AI Bottleneck Dashboard — {data_type.title()}")

    return {
        "bottlenecks": results,
        "ml_model_card": ml_card,
        "artifacts": {
            "dashboard_html": dash_path,
            "bottleneck_report_html": str(OUT_DIR / f"{data_type.lower()}_bottlenecks.html"),
            "ml_report_html": str(OUT_DIR / f"{data_type.lower()}_ml_report.html"),
            "static_urls": {
                "dashboard": f"/out/dashboard.html",
                "bottlenecks": f"/out/{data_type.lower()}_bottlenecks.html",
                "ml": f"/out/{data_type.lower()}_ml_report.html",
            }
        }
    }


@app.get("/whatif/{data_type}")
def what_if(
    data_type: str,
    traffic_multiplier: float = Query(1.0, ge=0.5, le=2.0),   # logistics
    latency_offset_ms: float = Query(0.0, ge=-50.0, le=200.0) # manufacturing
):
    
    data_type_u = data_type.upper()
    file_map = {
        "MANUFACTURING": PROJECT_ROOT / "manufacturing.csv",
        "LOGISTICS":     PROJECT_ROOT / "logistics.csv",
    }
    if data_type_u not in file_map:
        raise HTTPException(status_code=400, detail="data_type must be manufacturing or logistics")
    if not file_map[data_type_u].exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {file_map[data_type_u]}")

    base = _rows_from_csv(file_map[data_type_u], data_type_u)

    # clone rows and apply what-if
    import copy
    sim = copy.deepcopy(base)
    if data_type_u == "LOGISTICS":
        for r in sim:
            if "Delivery_Time" in r:
                try:
                    r["Delivery_Time"] = float(r["Delivery_Time"]) * traffic_multiplier
                except Exception:
                    pass
    else:  # MANUFACTURING
        for r in sim:
            if "Network_Latency_ms" in r:
                try:
                    r["Network_Latency_ms"] = float(r["Network_Latency_ms"]) + latency_offset_ms
                except Exception:
                    pass

    # run detections on baseline and simulated
    det_base = _detect_bottlenecks(base, data_type_u)
    det_sim  = _detect_bottlenecks(sim,  data_type_u)

    # compare severity per entity/type
    def key(row): return (row["entity"], row["bottleneck_type"])
    base_map = {key(r): r for r in det_base}
    sim_map  = {key(r): r for r in det_sim}
    delta = []
    for k in set(base_map) | set(sim_map):
        b = base_map.get(k)
        s = sim_map.get(k)
        sev_b = b.get("severity_score", 0.0) if b else 0.0
        sev_s = s.get("severity_score", 0.0) if s else 0.0
        delta.append({
            "entity": k[0],
            "bottleneck_type": k[1],
            "severity_before": round(sev_b,2),
            "severity_after":  round(sev_s,2),
            "delta":           round(sev_s - sev_b,2)
        })
    
    delta_sorted = sorted(delta, key=lambda d: -abs(d["delta"]))[:20]

    return {
        "params": {"traffic_multiplier": traffic_multiplier, "latency_offset_ms": latency_offset_ms},
        "delta": delta_sorted
    }


@app.get("/dashboard")
def get_dashboard():
    """
    Minimal DB-backed summary for the dashboard page.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        
        return {"error": f"DB connection failed: {e}", "bottlenecks": [], "performance": [], "distribution": [], "predictions": []}

    cur = conn.cursor()
    cur.execute(
        """
        SELECT location_identifier, severity_score, bottleneck_type
        FROM bottleneck_results
        ORDER BY detected_at DESC
        LIMIT 10
        """
    )
    bottlenecks = [{"entity": row[0], "metric_value": row[1], "type": row[2]} for row in cur.fetchall()]

    cur.execute("SELECT AVG(severity_score) FROM bottleneck_results")
    avg_sev = cur.fetchone()[0] or 0.0

    cur.close(); conn.close()
    return {
        "bottlenecks": bottlenecks,
        "performance": [{"metric": "avg_severity", "value": round(float(avg_sev), 3)}],
        "distribution": [],  
        "predictions": []    
    }



# Local run helper

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipelines.main:app", host="127.0.0.1", port=8000, reload=True)
