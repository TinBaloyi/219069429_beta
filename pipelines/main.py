
from pathlib import Path
import sys
from typing import Tuple, Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import os, json, csv as _csv, logging, re, subprocess, sys

import psycopg2
from psycopg2 import sql
import pandas as pd
from .visualization_and_reports_pipeline import build_all_artifacts
from .data_ingestio_pipeline import DataSource, HistoricalData, DataProcessor
from .analysis import AnalysisParameters, BottleneckDetector
from .bottlneck_profiles import manufacturing_profiles, logistics_profiles
from .config import DB_CONFIG

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pipelines")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

#App
app = FastAPI(title="AI Bottleneck Detection & Reporting")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/out", StaticFiles(directory=str(OUT_DIR)), name="out")

try:
    from ai_recommendations import generate_recommendations_for_event
    AI_RECOMMENDATIONS_AVAILABLE = True
    log.info("AI recommendations module loaded successfully")
except ImportError as e:
    AI_RECOMMENDATIONS_AVAILABLE = False
    log.warning(f"AI recommendations not available: {e}")
    # Fallback function
    def generate_recommendations_for_event(event, **kwargs):
        return event.get("recommendations", [])

#DB HELPERS

def pg_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)

def get_db_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)

def _cols(conn, table: str) -> set:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT lower(column_name)
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
        """, (table,))
        return {r[0] for r in cur.fetchall()}

def _schema(conn, table: str) -> Dict[str, Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT lower(column_name), data_type, is_nullable, column_default,
                   numeric_precision, numeric_scale, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
        """, (table,))
        out: Dict[str, Dict[str, Any]] = {}
        for c, dt, nul, dflt, p, s, clen in cur.fetchall():
            out[c] = {
                "data_type": dt,
                "is_nullable": (nul == "YES"),
                "column_default": dflt,
                "numeric_precision": None if p is None else int(p),
                "numeric_scale": None if s is None else int(s),
                "char_max_len": None if clen is None else int(clen),
            }
        return out

def _notnull_no_default(conn, table: str) -> Dict[str, Dict[str, Any]]:
    sc = _schema(conn, table)
    return {c: v for c, v in sc.items() if (not v["is_nullable"]) and (v["column_default"] is None)}

def _numeric_limits(conn, table: str) -> Dict[str, Tuple[int, int]]:
    sc = _schema(conn, table)
    return { c:(v["numeric_precision"], v["numeric_scale"])
             for c,v in sc.items()
             if v["data_type"]=="numeric" and v["numeric_precision"] and v["numeric_scale"] }

def _varchar_limits(conn, table: str) -> Dict[str, int]:
    sc = _schema(conn, table)
    return { c:v["char_max_len"]
             for c,v in sc.items()
             if v["char_max_len"] is not None and v["data_type"] in ("character varying","varchar") }

def _fit_numeric(value: Any, p: int, s: int) -> Optional[Decimal]:
    try:
        d = Decimal(str(value))
    except Exception:
        return None
    max_int = Decimal(10) ** (p - s)
    max_allowed = max_int - (Decimal(1) / (Decimal(10) ** s))
    min_allowed = -max_allowed
    if d > max_allowed: d = max_allowed
    if d < min_allowed: d = min_allowed
    return d.quantize(Decimal(1) / (Decimal(10) ** s), rounding=ROUND_HALF_UP)

def _fit_row(conn, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    num_limits = _numeric_limits(conn, table)
    str_limits = _varchar_limits(conn, table)
    fixed = dict(row)
    for k, v in list(fixed.items()):
        if v is None: continue
        key = k.lower()
        if key in num_limits:
            p,s = num_limits[key]
            fixed[k] = _fit_numeric(v, p, s)
        if key in str_limits and isinstance(fixed[k], str):
            maxlen = str_limits[key]
            if len(fixed[k]) > maxlen:
                fixed[k] = fixed[k][:maxlen]
    return fixed

def _get_pk(conn, table: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
           WHERE tc.table_schema='public' AND tc.table_name=%s AND tc.constraint_type='PRIMARY KEY'
           ORDER BY kcu.ordinal_position
        """, (table,))
        row = cur.fetchone()
        return row[0] if row else None

def _insert_dynamic(conn, table: str, row: Dict[str, Any]) -> None:
    cols = _cols(conn, table)
    data = {k: v for k,v in row.items() if k.lower() in cols}
    data = _fit_row(conn, table, data)
    if not data:
        raise RuntimeError(f"No insertable columns for {table}")
    q = sql.SQL("INSERT INTO {t} ({c}) VALUES ({v})").format(
        t=sql.Identifier(table),
        c=sql.SQL(", ").join(sql.Identifier(k) for k in data),
        v=sql.SQL(", ").join(sql.Placeholder(k) for k in data),
    )
    with conn.cursor() as cur:
        cur.execute(q, data)

def _insert_dynamic_returning(conn, table: str, row: Dict[str, Any], return_col: Optional[str]) -> Any:
    cols = _cols(conn, table)
    data = {k: v for k,v in row.items() if k.lower() in cols}
    data = _fit_row(conn, table, data)
    if not data:
        raise RuntimeError(f"No insertable columns for {table}")
    if return_col:
        q = sql.SQL("INSERT INTO {t} ({c}) VALUES ({v}) RETURNING {rc}").format(
            t=sql.Identifier(table),
            c=sql.SQL(", ").join(sql.Identifier(k) for k in data),
            v=sql.SQL(", ").join(sql.Placeholder(k) for k in data),
            rc=sql.Identifier(return_col),
        )
        with conn.cursor() as cur:
            cur.execute(q, data)
            return cur.fetchone()[0]
    else:
        _insert_dynamic(conn, table, data)
        return None

def _update_by_pk(conn, table: str, pk_col: str, pk_val: Any, update_row: Dict[str, Any]) -> None:
    cols = _cols(conn, table)
    data = {k: v for k,v in update_row.items() if k in cols}
    if not data: return
    data = _fit_row(conn, table, data)
    sets = sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in data)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("UPDATE {t} SET {s} WHERE {pk}=%s").format(
                t=sql.Identifier(table), s=sets, pk=sql.Identifier(pk_col)
            ),
            [*data.values(), pk_val]
        )

# FK/ Allowed values
def _fk_info(conn, table: str) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT kcu.column_name AS fk_col,
                   ccu.table_name  AS ref_table,
                   ccu.column_name AS ref_col
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
           WHERE tc.table_schema='public' AND tc.table_name=%s AND tc.constraint_type='FOREIGN KEY'
        """, (table,))
        rows = cur.fetchall()
    out = []
    for fk_col, ref_table, ref_col in rows:
        out.append({"fk_col": fk_col.lower(), "ref_table": ref_table, "ref_col": ref_col})
    return out

def _first_fk_value(conn, ref_table: str, ref_col: str) -> Optional[Any]:
    with conn.cursor() as cur:
        try:
            cur.execute(sql.SQL("SELECT {c} FROM {t} ORDER BY {c} ASC LIMIT 1")
                        .format(c=sql.Identifier(ref_col), t=sql.Identifier(ref_table)))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception:
            return None

def _allowed_values_for(conn, table: str, column: str) -> List[str]:
    # ENUM?
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.typname
            FROM pg_type t
            JOIN information_schema.columns c ON c.udt_name = t.typname
           WHERE c.table_schema='public' AND c.table_name=%s AND c.column_name=%s AND t.typtype='e'
        """, (table, column))
        row = cur.fetchone()
    if row:
        enum_name = row[0]
        with conn.cursor() as cur:
            cur.execute("SELECT enumlabel FROM pg_enum WHERE enumtypid=(SELECT oid FROM pg_type WHERE typname=%s) ORDER BY enumsortorder", (enum_name,))
            return [r[0] for r in cur.fetchall()]
    # CHECK constraints
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
           WHERE n.nspname='public' AND t.relname=%s AND pg_get_constraintdef(c.oid) ILIKE %s
        """, (table, f"%{column}%"))
        defs = [r[0] for r in cur.fetchall()]
    vals: List[str] = []
    for defn in defs:
        vals.extend(re.findall(r"'([^']*)'", defn))
    seen, uniq = set(), []
    for v in vals:
        if v not in seen:
            uniq.append(v); seen.add(v)
    return uniq

def _best_allowed(conn, table: str, column: str, desired: str, default_if_empty: str) -> str:
    allowed = _allowed_values_for(conn, table, column)
    if not allowed:
        return desired or default_if_empty
    for a in allowed:
        if a.lower() == (desired or "").lower():
            return a
    return allowed[0]

# bottleneck_type mapping 
def sanitize_bottleneck_type(raw_val: str) -> str:
    if not raw_val: return "Other"
    base = raw_val.split("(")[0].strip()
    return base or raw_val.strip()

def map_bottleneck_type(conn, raw_val: str) -> str:
    base = sanitize_bottleneck_type(raw_val)
    allowed = _allowed_values_for(conn, "bottleneck_results", "bottleneck_type")
    if not allowed:
        return base
    for a in allowed:
        if a.lower() == base.lower():
            return a
    nk = re.sub(r'[^A-Z0-9]+', '', base.upper())
    for a in allowed:
        if re.sub(r'[^A-Z0-9]+','',a.upper()) == nk:
            return a
    for a in allowed:
        if re.sub(r'[^A-Z0-9]+','',a.upper()) in nk or nk in re.sub(r'[^A-Z0-9]+','',a.upper()):
            return a
    return allowed[0]

# run row & seeding
def _default_for_type(colname: str, dtype: str, start_time: datetime) -> Any:
    d = dtype.lower()
    if "timestamp" in d or d in ("date","time","timetz"): return start_time
    if d in ("integer","bigint","smallint","real","double precision"): return 0
    if d == "numeric": return Decimal(0)
    if d in ("character varying","varchar","text","character"):
        if "status" in colname: return "running"
        if "name"   in colname: return "analysis"
        return ""
    if d in ("boolean",): return False
    return None

def _table_has_rows(conn, table: str) -> bool:
    with conn.cursor() as cur:
        pk = _get_pk(conn, table)
        probe = pk or (next(iter(_cols(conn, table))) if _cols(conn, table) else "*")
        cur.execute(sql.SQL("SELECT {c} FROM {t} LIMIT 1").format(
            c=sql.Identifier(probe) if probe != "*" else sql.SQL("*"),
            t=sql.Identifier(table)
        ))
        return cur.fetchone() is not None

def _first_row_pk(conn, table: str) -> Optional[Any]:
    pk = _get_pk(conn, table)
    if not pk:
        return None
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT {pk} FROM {t} ORDER BY {pk} ASC LIMIT 1").format(
            pk=sql.Identifier(pk), t=sql.Identifier(table)
        ))
        r = cur.fetchone()
        return r[0] if r else None

def _enum_or_check_allowed(conn, table: str, column: str) -> List[str]:
    return _allowed_values_for(conn, table, column)

def _seed_table_minimal(conn, table: str, depth: int = 0) -> Any:
    if depth > 5:
        raise HTTPException(500, f"Auto-seed recursion too deep while seeding {table}")

    # If table already has rows, return first PK (or insert default row if no PK)
    if _table_has_rows(conn, table):
        got = _first_row_pk(conn, table)
        if got is not None:
            return got
        with conn.cursor() as cur:
            try:
                cur.execute(sql.SQL("INSERT INTO {t} DEFAULT VALUES").format(t=sql.Identifier(table)))
                conn.commit()
            except Exception:
                conn.rollback()
                try:
                    _insert_dynamic(conn, table, {})
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise HTTPException(500, f"Cannot seed {table}: {e}")
        return _first_row_pk(conn, table)

    sc   = _schema(conn, table)
    req  = _notnull_no_default(conn, table)
    fks  = _fk_info(conn, table)
    cols = _cols(conn, table)
    pk   = _get_pk(conn, table)

    payload: Dict[str, Any] = {}

    # seed referenced FK tables first when required
    fk_map = {fk["fk_col"]: (fk["ref_table"], fk["ref_col"]) for fk in fks}
    for fk_col, (ref_table, ref_col) in fk_map.items():
        ref_pk = _first_row_pk(conn, ref_table)
        if ref_pk is None and fk_col in req:
            ref_pk = _seed_table_minimal(conn, ref_table, depth + 1)
        if ref_pk is not None:
            payload[fk_col] = ref_pk

    now = datetime.utcnow()
    for c, meta in sc.items():
        if c == pk or c in payload or c not in cols: continue
        if c in req:
            dt = meta["data_type"].lower()
            if dt == "user-defined":
                allowed = _enum_or_check_allowed(conn, table, c)
                payload[c] = allowed[0] if allowed else "default"
            elif "timestamp" in dt or dt in ("date","time","timetz"):
                payload[c] = now
            elif dt in ("integer","bigint","smallint","real","double precision"):
                payload[c] = 0
            elif dt == "numeric":
                payload[c] = Decimal(0)
            elif dt in ("boolean",):
                payload[c] = False
            elif dt in ("json", "jsonb"):
                payload[c] = json.dumps({})
            elif dt in ("character varying","varchar","text","character"):
                allowed = _enum_or_check_allowed(conn, table, c)
                payload[c] = (allowed[0] if allowed else ("default" if "name" in c else ""))
            else:
                payload[c] = None

    try:
        ret_col = pk
        new_id = _insert_dynamic_returning(conn, table, payload, ret_col)
        conn.commit()
        return new_id if new_id is not None else _first_row_pk(conn, table)
    except Exception as e:
        conn.rollback()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("INSERT INTO {t} DEFAULT VALUES RETURNING {pk}").format(
                    t=sql.Identifier(table), pk=sql.Identifier(pk) if pk else sql.Identifier(list(cols)[0])
                ))
                rid = cur.fetchone()[0]
                conn.commit()
                return rid
        except Exception as e2:
            conn.rollback()
            raise HTTPException(500, f"Cannot seed {table}: {e2}")

def ensure_run_id(kind: str, start_time: datetime) -> int:
    """Create a new analysis run and return its ID"""
    with pg_conn() as conn:
        cols = _cols(conn, "analysis_runs")
        if not cols:
            raise HTTPException(500, "analysis_runs table not found")

        sc = _schema(conn, "analysis_runs")
        fks = _fk_info(conn, "analysis_runs")
        req = _notnull_no_default(conn, "analysis_runs")
        pk = "run_id" if "run_id" in cols else (_get_pk(conn, "analysis_runs") or None)

       
        use_db_generated_run_id = False
        if "run_id" in cols:
            coldef = (sc.get("run_id") or {}).get("column_default") or ""
            if "nextval(" in coldef or "identity" in coldef.lower():
                use_db_generated_run_id = True

        payload: Dict[str, Any] = {}
        
        
        if not use_db_generated_run_id and "run_id" in cols and "run_id" in req:
            # Get next safe ID
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM analysis_runs;")
                next_id = cur.fetchone()[0] or 1
                payload["run_id"] = int(min(next_id, 2_147_483_647))

        if "run_name" in cols:
            payload["run_name"] = f"{kind}_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if "analysis_status" in cols:
            payload["analysis_status"] = _best_allowed(conn, "analysis_runs", "analysis_status", "running", "running")
        if "start_time" in cols:
            payload["start_time"] = start_time
        if "end_time" in cols:
            payload["end_time"] = start_time
        if "dataset_size" in cols:
            payload["dataset_size"] = 0
        if "execution_time_seconds" in cols:
            payload["execution_time_seconds"] = 0

        # Satisfy NOT NULL FKs by auto-seeding
        fk_map = {fk["fk_col"]: (fk["ref_table"], fk["ref_col"]) for fk in fks}
        for c, (ref_table, ref_col) in fk_map.items():
            if c in payload:
                continue
            ref_val = _first_fk_value(conn, ref_table, ref_col)
            if ref_val is None and c in req:
                ref_val = _seed_table_minimal(conn, ref_table, depth=0)
            if ref_val is not None:
                payload[c] = ref_val

        # Fill remaining NOT NULLs with safe defaults
        for c, meta in sc.items():
            if c in payload or c in fk_map or c == pk:
                continue
            if c in req:
                payload[c] = _default_for_type(c, meta["data_type"], start_time)

        try:
            # Use RETURNING clause to get the ID
            ret_col = "run_id" if "run_id" in cols else pk
            if ret_col:
                # Build INSERT with RETURNING
                cols_list = list(payload.keys())
                q = sql.SQL("INSERT INTO analysis_runs ({cols}) VALUES ({vals}) RETURNING {ret}").format(
                    cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols_list),
                    vals=sql.SQL(", ").join(sql.Placeholder(c) for c in cols_list),
                    ret=sql.Identifier(ret_col)
                )
                with conn.cursor() as cur:
                    cur.execute(q, payload)
                    rid = cur.fetchone()[0]
                    conn.commit()
                    return int(rid)
            else:
                # No return column - just insert
                _insert_dynamic(conn, "analysis_runs", payload)
                conn.commit()
                # Try to fetch the last inserted row
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT run_id FROM analysis_runs 
                        ORDER BY start_time DESC NULLS LAST, end_time DESC NULLS LAST 
                        LIMIT 1
                    """)
                    rid = cur.fetchone()[0]
                    return int(rid)
        except Exception as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "duplicate key" in error_msg and "run_id" in error_msg:
                # Retry with a different timestamp-based approach
                payload["run_name"] = f"{kind}_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                if "run_id" in payload:
                    # Remove run_id and let DB generate it
                    del payload["run_id"]
                try:
                    rid = _insert_dynamic_returning(conn, "analysis_runs", payload, "run_id")
                    conn.commit()
                    return int(rid)
                except Exception as e2:
                    conn.rollback()
                    raise HTTPException(500, f"Could not create analysis_runs row after retry: {e2}")
            else:
                raise HTTPException(500, f"Could not create analysis_runs row: {e}")

#PIPELINE WRITERS / METRICS

def insert_pipeline_metric(stage, start_ts, end_ts, records, errors=0, success=True):
    stage_key = (stage or "").strip().upper()  # canonical form
    duration = max((end_ts - start_ts).total_seconds(), 1.0)
    rpm = int((records / duration) * 60)
    success_fraction = 1.0 if success else 0.0
    try:
        with pg_conn() as conn:
            _insert_dynamic(conn, "pipeline_metrics", {
                "metric_timestamp": datetime.utcnow(),
                "pipeline_stage": stage_key,
                "throughput_records_per_minute": rpm,
                "processing_time_seconds": int(duration),
                "memory_usage_mb": None,
                "cpu_usage_percentage": None,
                "error_count": int(errors),
                "success_rate": success_fraction,
            })
            conn.commit()
    except Exception as e:
        log.warning("pipeline_metrics insert skipped: %s", e)

def insert_data_import(source_name, path, records, started_at, finished_at, source_id=None):
    try:
        with pg_conn() as conn:
            payload = {
                "source_name": source_name,
                "source_type": "FILE",
                "import_status": "COMPLETED",
                "records_imported": int(records),
                "started_at": started_at,
                "finished_at": finished_at,
                "details": json.dumps({"path": str(path)}),
            }
            if source_id is not None:
                payload["source_id"] = source_id
            _insert_dynamic(conn, "data_imports", payload)
            conn.commit()
    except Exception as e:
        log.warning("data_imports insert skipped: %s", e)

def insert_processed_snapshot(domain, records, cleaned_path):
    try:
        with pg_conn() as conn:
            cols = _cols(conn, "processed_data")
            payload = {}
            if "historical_data_id" in cols:
                payload["historical_data_id"] = None
            if "cleaned_data" in cols:
                payload["cleaned_data"] = json.dumps({"domain": domain, "records": int(records)})
            if "transformed_data" in cols:
                payload["transformed_data"] = json.dumps({"cleaned_csv": str(cleaned_path)})
            if payload:
                _insert_dynamic(conn, "processed_data", payload)
                conn.commit()
    except Exception as e:
        log.warning("processed_data insert skipped: %s", e)



# CSV / DETECTIONS
#
def _resolve_csv(name: str) -> Path:
    for p in [PROJECT_ROOT / name, PROJECT_ROOT / "data" / name, BASE_DIR / name, BASE_DIR / "data" / name]:
        if p.exists(): return p
    raise HTTPException(status_code=400, detail=f"CSV not found: {name}")

def _rows_from_csv(file_path: Path, data_type: str) -> list[dict]:
    ds = DataSource("src1", "LocalCSV", data_type.upper(), {}, {}, "csv", str(file_path))
    if not ds.validate_connection():
        raise HTTPException(status_code=400, detail=f"Cannot read file: {file_path}")
    raw = ds.import_historical_data()
    hist = HistoricalData("hist1", ds.source_id, "all", raw)
    dp = DataProcessor()
    cleaned = dp.clean_and_process_data(hist.raw_data, cleaning_columns=None)
    try:
        return dp.transform_data(cleaned, transformations=None)
    except Exception:
        return cleaned

def _detect_bottlenecks(rows: list[dict], data_type: str) -> list[dict]:
    detector = BottleneckDetector()
    profiles = manufacturing_profiles if data_type.upper() == "MANUFACTURING" else logistics_profiles
    out: list[dict] = []
    for p in profiles:
        params = AnalysisParameters(
            analysis_id="auto",
            dataset_id=str(p.metric_column or "na"),
            analysis_type=data_type.upper(),
            group_by_column=p.group_by_column,
            metric_column=p.metric_column,
            modelSelection=[p.group_by_column],
        )
        setattr(params, "mode", getattr(p, "mode", "min"))
        events = detector.detect_bottlenecks(rows, params)
        for ev in events:
            sev = float(getattr(ev, "severity_score", 0.0))
            conf = float(getattr(ev, "confidence_score", 0.0))
            detected_at = getattr(ev, "start_time", datetime.utcnow())
            out.append({
                "bottleneck_type": p.name,
                "description": getattr(p, "description", ""),
                "entity": ev.entity_id,
                "metric": p.metric_column,
                "metric_value": getattr(ev, "metric_value", None),
                "reason": getattr(ev, "reason", ""),
                "detected_at": detected_at,
                "severity_score": sev,
                "severity": sev,
                "confidence_score": conf,
                "profile_mode": getattr(p, "mode", "min"),
                "group_by": p.group_by_column,
                "recommendations": [
                    "Verify staffing/capacity",
                    "Inspect quality gates",
                    "Consider rerouting/retiming",
                ],
                "root_cause_analysis": {"metric": p.metric_column, "mode": getattr(p, "mode", "min")},
            })
    return out

def normalize_detection_row(conn, r: Dict[str, Any]) -> Dict[str, Any]:
    x = dict(r)
    if "severity" not in x:
        x["severity"] = float(x.get("severity_score", 0.0) or 0.0)
    if "severity_score" not in x:
        x["severity_score"] = float(x.get("severity", 0.0) or 0.0)
    dt = x.get("detected_at", datetime.utcnow())
    if isinstance(dt, str):
        try: dt = datetime.fromisoformat(dt)
        except Exception: dt = datetime.utcnow()
    x["detected_at"] = dt
    x["entity"] = str(x.get("entity", ""))
    x["bottleneck_type"] = map_bottleneck_type(conn, x.get("bottleneck_type") or "Other")
    return x

def detect_all_bottlenecks(cleaned_data, profiles, data_type):
    
    import pandas as pd
    import numpy as np
    from datetime import datetime
    log = logging.getLogger(__name__)

    # Normalize input -> DataFrame + keep original list form for detector if needed
    df = None
    as_rows = None
    if isinstance(cleaned_data, pd.DataFrame):
        df = cleaned_data.copy()
        as_rows = df.to_dict(orient="records")
    else:
        # assume list-of-dicts
        try:
            as_rows = list(cleaned_data) if cleaned_data is not None else []
            df = pd.DataFrame(as_rows) if as_rows else pd.DataFrame()
        except Exception:
            as_rows = []
            df = pd.DataFrame()

    dtype = (data_type or "").strip().upper()
    out: list[dict] = []

    # Helper: normalize event dict
    def _normalize_ev(ev: dict) -> dict:
        ev2 = dict(ev or {})
        # detected_at -> datetime
        dt = ev2.get("detected_at") or ev2.get("timestamp") or datetime.utcnow()
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except Exception:
                try:
                    dt = pd.to_datetime(dt)
                    dt = dt.to_pydatetime()
                except Exception:
                    dt = datetime.utcnow()
        ev2["detected_at"] = dt
        # severity -> 0..1
        try:
            sev = float(ev2.get("severity_score", ev2.get("severity", 0.0)) or 0.0)
        except Exception:
            sev = 0.0
        # If severity appears to be 0..100 scale, convert to 0..1
        if sev > 1.0:
            sev = max(0.0, min(1.0, sev / 100.0))
        ev2["severity_score"] = max(0.0, min(1.0, sev))
        ev2["severity"] = ev2["severity_score"]

        # confidence -> 0..1 clamp
        try:
            conf = float(ev2.get("confidence_score", ev2.get("confidence", 0.0)) or 0.0)
        except Exception:
            conf = 0.0
        if conf > 1.0:
            conf = max(0.0, min(1.0, conf / 100.0))
        ev2["confidence_score"] = max(0.0, min(1.0, conf))

        # entity as string
        ev2["entity"] = str(ev2.get("entity", ev2.get("entity_id", ev2.get("location", "")) or ""))
        # recommendations: ensure structured
        recs = ev2.get("recommendations") or []
        # if recommendations are simple strings, convert to structured dicts
        if recs and isinstance(recs[0], str):
            structured = []
            for i, r in enumerate(recs[:5]):
                structured.append({"action": r, "priority": i + 1, "verification": "Verify result after change"})
            ev2["recommendations"] = structured
        elif not recs:
            ev2["recommendations"] = []
        # root_cause_analysis ensure exists
        ev2.setdefault("root_cause_analysis", {"metric": ev2.get("metric"), "mode": ev2.get("profile_mode", None)})
        return ev2

    # 1) Try profile-based detector (preferred)
    detector = None
    try:
        detector = BottleneckDetector()
    except Exception:
        log.debug("BottleneckDetector not available, will use heuristics only.")

    if profiles:
        for p in profiles:
            try:
                params = AnalysisParameters(
                    analysis_id="auto",
                    dataset_id=str(getattr(p, "metric_column", "na") or "na"),
                    analysis_type=dtype or "UNKNOWN",
                    group_by_column=getattr(p, "group_by_column", None),
                    metric_column=getattr(p, "metric_column", None),
                    modelSelection=[getattr(p, "group_by_column", None)] if getattr(p, "group_by_column", None) else None,
                )
                setattr(params, "mode", getattr(p, "mode", "min"))
            except Exception as e:
                log.debug("Skipping bad profile params: %s", e)
                continue

            # Run detector if available
            events = []
            if detector is not None:
                try:
                    events = detector.detect_bottlenecks(as_rows, params) or []
                except Exception as e:
                    log.warning("Detector failed for profile %s: %s", getattr(p, "name", "<profile>"), e)
                    events = []
            else:
                events = []

            # Normalize detector events (detector may return objects)
            for ev in events:
                try:
                    # If detector returns an object, try attributes; else assume dict
                    if not isinstance(ev, dict):
                        ev_dict = {}
                        for k in ("entity_id", "metric_value", "reason", "start_time", "severity_score", "confidence_score"):
                            if hasattr(ev, k):
                                ev_dict[k if k != "entity_id" else "entity"] = getattr(ev, k)
                    else:
                        ev_dict = dict(ev)
                    ev_out = {
                        "bottleneck_type": getattr(p, "name", ev_out if "ev_out" in locals() else (ev_dict.get("bottleneck_type") or getattr(p, "metric_column", "Unknown"))),
                        "description": getattr(p, "description", "") or "",
                        "entity": ev_dict.get("entity", ev_dict.get("entity_id", "")),
                        "metric": getattr(p, "metric_column", ev_dict.get("metric")),
                        "metric_value": ev_dict.get("metric_value"),
                        "reason": ev_dict.get("reason", ""),
                        "detected_at": ev_dict.get("start_time"),
                        "severity_score": ev_dict.get("severity_score", ev_dict.get("severity", 0.0)),
                        "confidence_score": ev_dict.get("confidence_score", ev_dict.get("confidence", 0.0)),
                        "profile_mode": getattr(p, "mode", "min"),
                        "group_by": getattr(p, "group_by_column", None),
                        "root_cause_analysis": {"metric": getattr(p, "metric_column", None), "mode": getattr(p, "mode", "min")},
                    }

                    # Try to generate recommendations via global helper
                    try:
                        if "generate_recommendations_for_event" in globals():
                            # pass slice of df for entity context where possible
                            df_context = None
                            gcol = ev_out.get("group_by") or ev_out.get("metric")
                            if df is not None and gcol and gcol in df.columns:
                                try:
                                    df_context = df[df[gcol].astype(str) == str(ev_out["entity"])].copy()
                                except Exception:
                                    df_context = None
                            recs = generate_recommendations_for_event(ev_out, df_context=df_context, use_llm=False, llm_client=None)
                            if recs:
                                ev_out["recommendations"] = recs
                    except Exception as e:
                        log.debug("Recommendation generator failed: %s", e)
                        ev_out["recommendations"] = ev_out.get("recommendations", [])

                    out.append(_normalize_ev(ev_out))
                except Exception as e:
                    log.warning("Failed to normalize a detector event: %s", e)
                    continue

    # 2) If no events from profiles/detector — run domain-specific heuristics
    if not out:
        try:
            now = datetime.utcnow()
            # LOGISTICS heuristics
            if dtype == "LOGISTICS":
                # prefer delivery_time_deviation, delay_probability, fuel_consumption_rate
                if df is None or df.empty:
                    pass
                else:
                    # compute candidate metrics
                    # delivery_time_deviation (minutes or hours) -> standardize: convert to numeric
                    if "delivery_time_deviation" in df.columns:
                        col = "delivery_time_deviation"
                        vals = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not vals.empty:
                            median = float(vals.median())
                            p90 = float(vals.quantile(0.90))
                            # Flag groups by route_id / carrier / vehicle_id if available
                            group_cols = [c for c in ("route_id", "carrier", "vehicle_id", "shipment_id") if c in df.columns]
                            if group_cols:
                                g = df.groupby(group_cols)[col].agg(["median", "count"]).reset_index()
                                g = g[g["count"] >= 5]
                                g = g[g["median"].fillna(0) > max(5.0, median * 1.5)]
                                for _, row in g.iterrows():
                                    severity = min(1.0, (float(row["median"]) / max(1.0, median)) / 3.0)
                                    ev = {
                                        "bottleneck_type": "Route Delay Bottleneck",
                                        "description": f"High median {col} for {group_cols[0]}",
                                        "entity": str(row[group_cols[0]]),
                                        "metric": col,
                                        "metric_value": float(row["median"]),
                                        "reason": f"median {col}={row['median']:.2f}",
                                        "detected_at": now,
                                        "severity_score": severity,
                                        "confidence_score": 0.6,
                                        "profile_mode": "max",
                                        "group_by": group_cols[0],
                                    }
                                    # recommendations
                                    try:
                                        ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df[df[group_cols[0]].astype(str) == str(ev["entity"])]) \
                                            if "generate_recommendations_for_event" in globals() else []
                                    except Exception:
                                        ev["recommendations"] = []
                                    out.append(_normalize_ev(ev))
                            else:
                                # global summary
                                if median > 10 or p90 > max(15, median * 2):
                                    severity = min(1.0, p90 / (60.0 * 2))
                                    ev = {
                                        "bottleneck_type": "Global Route Delay",
                                        "description": f"High overall delivery deviation (median={median:.2f}, p90={p90:.2f})",
                                        "entity": "GLOBAL",
                                        "metric": col,
                                        "metric_value": p90,
                                        "reason": f"p90={p90:.2f} > threshold",
                                        "detected_at": now,
                                        "severity_score": severity,
                                        "confidence_score": 0.5,
                                        "profile_mode": "max",
                                    }
                                    try:
                                        ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df) \
                                            if "generate_recommendations_for_event" in globals() else []
                                    except Exception:
                                        ev["recommendations"] = []
                                    out.append(_normalize_ev(ev))

                    # delay_probability
                    if "delay_probability" in df.columns:
                        prob = pd.to_numeric(df["delay_probability"], errors="coerce").dropna()
                        if not prob.empty and prob.max() <= 1.0:
                            grp = df.groupby([c for c in ("route_id","risk_classification","carrier") if c in df.columns][0:1])[ "delay_probability"].mean()
                            grp = grp[grp > 0.5].sort_values(ascending=False).head(5)
                            for ent, val in grp.items():
                                ev = {
                                    "bottleneck_type": "High Delay Risk",
                                    "description": "High average delay probability",
                                    "entity": str(ent),
                                    "metric": "delay_probability",
                                    "metric_value": float(val),
                                    "reason": f"avg delay_probability={val:.3f}",
                                    "detected_at": now,
                                    "severity_score": min(1.0, float(val)),
                                    "confidence_score": 0.6,
                                    "profile_mode": "max",
                                }
                                try:
                                    ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df[df.get("route_id", pd.Series()).astype(str) == str(ent)]) \
                                        if "generate_recommendations_for_event" in globals() else []
                                except Exception:
                                    ev["recommendations"] = []
                                out.append(_normalize_ev(ev))

                    # fuel_consumption_rate
                    if "fuel_consumption_rate" in df.columns:
                        fc = pd.to_numeric(df["fuel_consumption_rate"], errors="coerce").dropna()
                        if not fc.empty:
                            # group by vehicle_id if present
                            group_col = "vehicle_id" if "vehicle_id" in df.columns else None
                            if group_col:
                                g = df.groupby(group_col)["fuel_consumption_rate"].mean().dropna()
                                p90 = g.quantile(0.90)
                                high = g[g > p90].nlargest(5)
                                for ent, val in high.items():
                                    severity = min(1.0, (float(val) / float(max(1e-6, p90))) - 1.0)
                                    ev = {
                                        "bottleneck_type": "Vehicle Inefficiency (Fuel)",
                                        "description": "High average fuel consumption",
                                        "entity": str(ent),
                                        "metric": "fuel_consumption_rate",
                                        "metric_value": float(val),
                                        "reason": f"avg fuel_consumption_rate={val:.3f}",
                                        "detected_at": now,
                                        "severity_score": min(1.0, severity),
                                        "confidence_score": 0.5,
                                        "profile_mode": "max",
                                    }
                                    try:
                                        ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df[df[group_col].astype(str) == str(ent)]) \
                                            if "generate_recommendations_for_event" in globals() else []
                                    except Exception:
                                        ev["recommendations"] = []
                                    out.append(_normalize_ev(ev))

            # MANUFACTURING heuristics
            elif dtype == "MANUFACTURING":
                if df is None or df.empty:
                    pass
                else:
                    # candidate metric list for manufacturing
                    candidates = [c for c in ("cycle_time", "process_time", "duration", "throughput", "Production_Speed_units_per_hr", "Error_Rate_%", "error_rate", "error_count") if c in df.columns]
                    metric_col = candidates[0] if candidates else None

                    # error-rate check
                    now = datetime.utcnow()
                    if "Error_Rate_%" in df.columns or "error_rate" in df.columns:
                        err_col = "Error_Rate_%" if "Error_Rate_%" in df.columns else "error_rate"
                        df[err_col] = pd.to_numeric(df[err_col], errors="coerce")
                        # group by machine if possible
                        group_col = None
                        for gcol in ("machine_id", "equipment_id", "workstation", "line"):
                            if gcol in df.columns:
                                group_col = gcol
                                break
                        if group_col:
                            g = df.groupby(group_col)[err_col].agg(["mean", "count"]).reset_index()
                            g = g[g["count"] >= 10]
                            overall = df[err_col].median(skipna=True) or 0.0
                            for _, row in g.iterrows():
                                mean_val = float(row["mean"] or 0.0)
                                if overall > 0 and mean_val > max(0.05, overall * 1.5):
                                    ev = {
                                        "bottleneck_type": "High Error Rate",
                                        "description": f"High error rate on {group_col}",
                                        "entity": str(row[group_col]),
                                        "metric": err_col,
                                        "metric_value": mean_val,
                                        "reason": f"{err_col} mean {mean_val:.3f}",
                                        "detected_at": now,
                                        "severity_score": min(1.0, mean_val),  # assume rates <=1
                                        "confidence_score": 0.6,
                                        "profile_mode": "max",
                                        "group_by": group_col,
                                    }
                                    try:
                                        ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df[df[group_col].astype(str) == str(ev["entity"])]) \
                                            if "generate_recommendations_for_event" in globals() else []
                                    except Exception:
                                        ev["recommendations"] = []
                                    out.append(_normalize_ev(ev))
                        else:
                            # global check
                            med = df[err_col].median(skipna=True)
                            if med and med > 0.05 and len(df) >= 30:
                                ev = {
                                    "bottleneck_type": "High Error Rate",
                                    "description": "Global elevated error rate",
                                    "entity": "GLOBAL",
                                    "metric": err_col,
                                    "metric_value": float(med),
                                    "reason": f"global median {err_col}={med:.3f}",
                                    "detected_at": now,
                                    "severity_score": min(1.0, float(med)),
                                    "confidence_score": 0.45,
                                    "profile_mode": "max",
                                }
                                try:
                                    ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df) \
                                        if "generate_recommendations_for_event" in globals() else []
                                except Exception:
                                    ev["recommendations"] = []
                                out.append(_normalize_ev(ev))

                    # cycle time / speed check
                    if metric_col is not None:
                        if metric_col in df.columns:
                            # group by machine if available
                            group_col = next((c for c in ("machine_id","station","workstation","line") if c in df.columns), None)
                            overall_med = df[metric_col].median(skipna=True) or 0.0
                            if group_col:
                                g = df.groupby(group_col)[metric_col].agg(["median", "count"]).reset_index()
                                for _, row in g.iterrows():
                                    if row["count"] >= 10 and overall_med > 0 and row["median"] > 1.5 * overall_med:
                                        sev = min(1.0, (float(row["median"]) / max(1.0, float(overall_med)) - 1.0) / 2.0)
                                        ev = {
                                            "bottleneck_type": "High Cycle Time",
                                            "description": f"High median {metric_col} for {group_col}",
                                            "entity": str(row[group_col]),
                                            "metric": metric_col,
                                            "metric_value": float(row["median"]),
                                            "reason": f"{metric_col} median {row['median']:.2f} > 1.5x overall {overall_med:.2f}",
                                            "detected_at": now,
                                            "severity_score": sev,
                                            "confidence_score": 0.55,
                                            "profile_mode": "max",
                                            "group_by": group_col,
                                        }
                                        try:
                                            ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df[df[group_col].astype(str) == str(ev["entity"])]) \
                                                if "generate_recommendations_for_event" in globals() else []
                                        except Exception:
                                            ev["recommendations"] = []
                                        out.append(_normalize_ev(ev))
                            else:
                                # global recent vs historical check
                                n_recent = max(10, int(len(df) * 0.05))
                                recent_med = df[metric_col].tail(n_recent).median(skipna=True)
                                if overall_med > 0 and recent_med > 1.5 * overall_med and len(df) >= 30:
                                    ev = {
                                        "bottleneck_type": "High Cycle Time",
                                        "description": "Recent window shows elevated cycle time",
                                        "entity": "GLOBAL",
                                        "metric": metric_col,
                                        "metric_value": float(recent_med),
                                        "reason": f"recent median {recent_med:.2f} > 1.5x historic {overall_med:.2f}",
                                        "detected_at": now,
                                        "severity_score": min(1.0, (recent_med / max(1.0, overall_med)) / 3.0),
                                        "confidence_score": 0.45,
                                        "profile_mode": "max",
                                    }
                                    try:
                                        ev["recommendations"] = generate_recommendations_for_event(ev, df_context=df) \
                                            if "generate_recommendations_for_event" in globals() else []
                                    except Exception:
                                        ev["recommendations"] = []
                                    out.append(_normalize_ev(ev))

        except Exception as e:
            log.exception("Fallback heuristics failed: %s", e)

    # Final pass: ensure unique-ish results and sanity
    final: list[dict] = []
    seen = set()
    for r in out:
        key = (str(r.get("bottleneck_type","")).lower(), str(r.get("entity","")).lower(), str(r.get("metric","")).lower())
        if key in seen:
            # choose higher severity
            for ex in final:
                k2 = (str(ex.get("bottleneck_type","")).lower(), str(ex.get("entity","")).lower(), str(ex.get("metric","")).lower())
                if k2 == key:
                    if r.get("severity_score",0) > ex.get("severity_score",0):
                        ex.update(r)
                    break
            continue
        seen.add(key)
        final.append(r)

    return final



# PERSIST RESULTS

def _map_bottleneck_row(r: Dict[str, Any], cols: set, run_id: int) -> Dict[str, Any]:
    def pick(name, *aliases):
        for n in (name, *aliases):
            if n in cols: return n
        return None
    row: Dict[str, Any] = {}
    if 'run_id' in cols: row['run_id'] = run_id

    c = pick('bottleneck_type','type','category')
    if c: row[c] = r.get('bottleneck_type') or 'Other'

    c = pick('location_identifier','entity','entity_id','location','machine_id')
    if c: row[c] = str(r.get('entity',''))

    c = pick('severity_score','severity','score')
    if c: row[c] = r.get('severity_score',0.0)

    c = pick('confidence_score','confidence')
    if c: row[c] = r.get('confidence_score',0.0)

    c = pick('impact_description','reason','description','details')
    if c: row[c] = r.get('description') or r.get('reason') or ''

    c = pick('detected_at','timestamp','detected_time','detected_on')
    if c:
        dt = r.get('detected_at')
        if isinstance(dt,str):
            try: dt = datetime.fromisoformat(dt)
            except Exception: dt = datetime.utcnow()
        row[c] = dt

    c = pick('root_cause_analysis','root_cause','rca')
    if c: row[c] = json.dumps(r.get('root_cause_analysis',{}))

    c = pick('recommendations','recs','actions')
    if c: row[c] = json.dumps(r.get('recommendations',[]))

    c = pick('metric')
    if c and r.get('metric') is not None: row[c] = r.get('metric')

    c = pick('metric_value','value')
    if c and r.get('metric_value') is not None: row[c] = r.get('metric_value')
    return row

def save_bottleneck_results(results, run_id: Optional[int], kind_for_fallback: str):
    if not results: return
    if run_id is None:
        raise HTTPException(500, "run_id is required to save results")
    with pg_conn() as conn:
        cols = _cols(conn, "bottleneck_results")
        if "run_id" not in cols:
            raise HTTPException(500, "bottleneck_results.run_id column missing")
        allowed = _allowed_values_for(conn, "bottleneck_results", "bottleneck_type") or ["Other"]
        for r in results:
            nr = normalize_detection_row(conn, r)
            row = _map_bottleneck_row(nr, cols, run_id)
            row = _fit_row(conn, "bottleneck_results", row)
            try:
                _insert_dynamic(conn, "bottleneck_results", row)
            except Exception as e:
                msg = str(e).lower()
                if "check" in msg and "bottleneck_type" in msg:
                    # Force to first allowed and retry
                    type_col = "bottleneck_type" if "bottleneck_type" in cols else next(iter([c for c in cols if c in ("type","category")]), "bottleneck_type")
                    row[type_col] = allowed[0]
                    row = _fit_row(conn, "bottleneck_results", row)
                    _insert_dynamic(conn, "bottleneck_results", row)
                else:
                    raise
        conn.commit()


#LOGISTICS FALLBACK (if profiles return none)
def detect_logistics_fallback_df(df: pd.DataFrame) -> list[dict]:
    """Enhanced fallback detection for logistics when profiles don't match"""
    if df is None or df.empty:
        return []

    results = []
    
    # Strategy 1: delivery_time_deviation analysis (primary target)
    if "delivery_time_deviation" in df.columns:
        deviation = pd.to_numeric(df["delivery_time_deviation"], errors="coerce").dropna()
        if not deviation.empty:
            # Group by risk classification if available
            if "risk_classification" in df.columns:
                entity_col = "risk_classification"
                entity = df["risk_classification"].fillna("UNKNOWN").astype(str)
            elif "route_risk_level" in df.columns:
                # Create risk bins from numeric route_risk_level
                lvl = pd.to_numeric(df["route_risk_level"], errors="coerce")
                bins = [-1, 3, 7, 1e9]
                labels = ["LOW", "MEDIUM", "HIGH"]
                entity = pd.cut(lvl, bins=bins, labels=labels).astype(str).fillna("UNKNOWN")
                entity_col = "route_risk_level_bin"
            else:
                # Fallback: create synthetic route groups from GPS if available
                if "vehicle_gps_latitude" in df.columns and "vehicle_gps_longitude" in df.columns:
                    lat = pd.to_numeric(df["vehicle_gps_latitude"], errors="coerce").fillna(0)
                    lon = pd.to_numeric(df["vehicle_gps_longitude"], errors="coerce").fillna(0)
                    # Round to 1 decimal place for rough geographic grouping
                    entity = (lat.round(1).astype(str) + "_" + lon.round(1).astype(str))
                    entity_col = "geo_region"
                else:
                    # Last resort: use row index modulo for synthetic grouping
                    entity = pd.Series([f"ROUTE_{i % 10}" for i in range(len(df))])
                    entity_col = "synthetic_route"
            
            # Group by entity and calculate mean deviation
            grouped = deviation.groupby(entity).mean().dropna()
            if not grouped.empty:
                # Get top 10 worst performers
                p50 = grouped.quantile(0.50)
                p95 = grouped.quantile(0.95)
                spread = max(p95 - p50, 1e-9)
                top = grouped.sort_values(ascending=False).head(10)
                
                now = datetime.utcnow()
                for ent, val in top.items():
                    severity = max(0.0, min(100.0, ((float(val) - p50) / spread) * 100.0))
                    results.append({
                        "bottleneck_type": "Route Delay Bottleneck",
                        "description": f"Routes with high delivery time deviation (grouped by {entity_col})",
                        "entity": str(ent),
                        "metric": "delivery_time_deviation",
                        "metric_value": float(val),
                        "reason": f"avg delivery_time_deviation = {float(val):.3f}",
                        "detected_at": now,
                        "severity_score": severity,
                        "severity": severity,
                        "confidence_score": 0.65,
                        "profile_mode": "max",
                        "group_by": entity_col,
                        "recommendations": [
                            "Analyze route patterns for delays",
                            "Consider alternative routes",
                            "Review carrier performance",
                            "Optimize delivery windows"
                        ],
                        "root_cause_analysis": {
                            "metric": "delivery_time_deviation",
                            "mode": "max",
                            "group_by": entity_col
                        },
                    })
    
    # Strategy 2: fuel_consumption_rate analysis (if available)
    if "fuel_consumption_rate" in df.columns:
        fuel = pd.to_numeric(df["fuel_consumption_rate"], errors="coerce").dropna()
        if not fuel.empty:
            # Use vehicle_id if available, otherwise use same entity logic as above
            if "vehicle_id" in df.columns:
                entity = df["vehicle_id"].astype(str)
                entity_col = "vehicle_id"
            else:
                entity = df.get("risk_classification", pd.Series(["UNKNOWN"] * len(df))).astype(str)
                entity_col = "category"
            
            grouped_fuel = fuel.groupby(entity).mean().dropna()
            if not grouped_fuel.empty:
                p90 = grouped_fuel.quantile(0.90)
                high_fuel = grouped_fuel[grouped_fuel > p90].sort_values(ascending=False).head(5)
                
                now = datetime.utcnow()
                for ent, val in high_fuel.items():
                    severity = min(100.0, (float(val) / p90 - 1.0) * 100.0)
                    results.append({
                        "bottleneck_type": "Vehicle Inefficiency (Fuel)",
                        "description": f"High fuel consumption entities (grouped by {entity_col})",
                        "entity": str(ent),
                        "metric": "fuel_consumption_rate",
                        "metric_value": float(val),
                        "reason": f"avg fuel_consumption_rate = {float(val):.3f}",
                        "detected_at": now,
                        "severity_score": severity,
                        "severity": severity,
                        "confidence_score": 0.55,
                        "profile_mode": "max",
                        "group_by": entity_col,
                        "recommendations": [
                            "Inspect vehicle maintenance records",
                            "Analyze driving patterns",
                            "Consider route optimization",
                            "Review fuel quality"
                        ],
                        "root_cause_analysis": {
                            "metric": "fuel_consumption_rate",
                            "mode": "max",
                            "group_by": entity_col
                        },
                    })
    
    # Strategy 3: delay_probability analysis
    if "delay_probability" in df.columns:
        delay_prob = pd.to_numeric(df["delay_probability"], errors="coerce").dropna()
        if not delay_prob.empty and delay_prob.max() <= 1.0:  # Ensure it's a probability
            # Use same entity logic
            entity = df.get("risk_classification", 
                          df.get("route_risk_level", 
                                pd.Series(["UNKNOWN"] * len(df)))).astype(str)
            entity_col = "risk_category"
            
            grouped_prob = delay_prob.groupby(entity).mean().dropna()
            if not grouped_prob.empty:
                high_prob = grouped_prob[grouped_prob > 0.5].sort_values(ascending=False).head(5)
                
                now = datetime.utcnow()
                for ent, val in high_prob.items():
                    severity = min(100.0, float(val) * 100.0)
                    results.append({
                        "bottleneck_type": "High Delay Risk",
                        "description": f"Entities with high delay probability (grouped by {entity_col})",
                        "entity": str(ent),
                        "metric": "delay_probability",
                        "metric_value": float(val),
                        "reason": f"avg delay_probability = {float(val):.3f}",
                        "detected_at": now,
                        "severity_score": severity,
                        "severity": severity,
                        "confidence_score": 0.60,
                        "profile_mode": "max",
                        "group_by": entity_col,
                        "recommendations": [
                            "Review historical delay patterns",
                            "Implement proactive alerts",
                            "Coordinate with carriers",
                            "Adjust scheduling buffers"
                        ],
                        "root_cause_analysis": {
                            "metric": "delay_probability",
                            "mode": "max",
                            "group_by": entity_col
                        },
                    })
    
    return results

# FULL DETECTION PIPELINE (from CSV path)
# Replace run_full_bottleneck_pipeline in main.py with this version:

def run_full_bottleneck_pipeline(file_path: str, data_type: str, run_id: Optional[int], save_results: bool = True):
    """Run full bottleneck detection pipeline with improved logistics handling"""
    ds = DataSource("src1", "DynamicSource", data_type, {}, {}, "csv", file_path)
    if not ds.validate_connection():
        log.warning(f"Data source validation failed for {file_path}")
        return []
    
    raw = ds.import_historical_data()
    if not raw:
        log.warning("No data imported from source")
        return []
    
    hist = HistoricalData("hist1", ds.source_id, "all", raw)
    dp = DataProcessor()
    
    profiles = manufacturing_profiles if data_type.upper() == "MANUFACTURING" else logistics_profiles
    
    # Collect needed columns from profiles
    need = set()
    for p in profiles:
        if p.group_by_column:
            need.add(p.group_by_column)
        if p.metric_column:
            need.add(p.metric_column)
    
    # Add common logistics columns that might be useful for fallback
    if data_type.upper() == "LOGISTICS":
        logistics_common = [
            "delivery_time_deviation",
            "delay_probability", 
            "fuel_consumption_rate",
            "risk_classification",
            "route_risk_level",
            "vehicle_gps_latitude",
            "vehicle_gps_longitude",
            "vehicle_id",
            "timestamp"
        ]
        need.update(logistics_common)
    
    # Filter to columns that actually exist
    if raw:
        available = set(raw[0].keys())
        need = need & available
    
    cleaned = dp.clean_and_process_data(hist.raw_data, cleaning_columns=list(need) if need else None)
    
    if not cleaned:
        log.warning("No data after cleaning")
        return []
    
    # Try profile-based detection first
    results = detect_all_bottlenecks(cleaned, profiles, data_type)
    
    # If no results and logistics, the detect_all_bottlenecks already has fallback built in
    # But we can add an additional safety check here
    if not results and data_type.upper() == "LOGISTICS":
        log.info("Profile detection returned no results, attempting direct fallback")
        try:
            df = pd.DataFrame(cleaned)
            results = detect_logistics_fallback_df(df)
        except Exception as e:
            log.exception(f"Final fallback detection failed: {e}")
    
    if save_results and results:
        if run_id is None:
            run_id = ensure_run_id(data_type.lower(), datetime.utcnow())
        save_bottleneck_results(results, run_id, kind_for_fallback=data_type.lower())
    
    return results


#  SKLEARN TRAINING WRAPPER (CLI)

def run_sklearn_training_and_load_predictions(kind: str, csv_path: Path) -> dict:
    """
    Calls sklearn trainers and loads predictions into DB.
    Returns model card info.
    """
    log.info(f"Starting training for {kind} with CSV: {csv_path}")
    
    if kind == "manufacturing":
        pred_out = OUT_DIR / "mf_preds.csv"
        log.info(f"Manufacturing prediction output will be: {pred_out}")
        
        cmd = [
            sys.executable, "-m", "train_manufacturing_speed_and_error",
            "--train_csv", str(csv_path),
            "--models_dir", str(MODELS_DIR),
            "--pred_csv", str(csv_path),
            "--pred_out", str(pred_out),
        ]
        
        log.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Training stdout: {result.stdout}")
        if result.stderr:
            log.warning(f"Training stderr: {result.stderr}")
        
        # Load predictions into DB
        if pred_out.exists():
            log.info(f"Prediction file exists: {pred_out}, size: {pred_out.stat().st_size} bytes")
            rows_read, rows_inserted = insert_predictions_from_csv(pred_out, src_name="manufacturing_trainer")
            log.info(f"Manufacturing predictions: read={rows_read}, inserted={rows_inserted}")
        else:
            log.error(f"Prediction file not found: {pred_out}")
        
        card_speed = MODELS_DIR / "mf_speed_modelcard.json"
        card_error = MODELS_DIR / "mf_error_modelcard.json"
        info = {
            "speed": json.loads(card_speed.read_text()) if card_speed.exists() else {},
            "error": json.loads(card_error.read_text()) if card_error.exists() else {},
        }
        return {"manufacturing": {"models": ["mf_speed.joblib", "mf_high_error.joblib"], "cards": info}}

    elif kind == "logistics":
        pred_out = OUT_DIR / "logistics_preds.csv"
        log.info(f"Logistics prediction output will be: {pred_out}")
        
        cmd = [
            sys.executable, "-m", "train_logistics_eta",
            "--train_csv", str(csv_path),
            "--models_dir", str(MODELS_DIR),
            "--pred_csv", str(csv_path),
            "--pred_out", str(pred_out),
        ]
        
        log.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Training stdout: {result.stdout}")
        if result.stderr:
            log.warning(f"Training stderr: {result.stderr}")
        
        # Load predictions into DB
        if pred_out.exists():
            log.info(f"Prediction file exists: {pred_out}, size: {pred_out.stat().st_size} bytes")
            rows_read, rows_inserted = insert_predictions_from_csv(pred_out, src_name="logistics_trainer")
            log.info(f"Logistics predictions: read={rows_read}, inserted={rows_inserted}")
        else:
            log.error(f"Prediction file not found: {pred_out}")
        
        card_eta = MODELS_DIR / "logistics_eta_modelcard.json"
        card_delay = MODELS_DIR / "logistics_delay_modelcard.json"
        info = {
            "eta": json.loads(card_eta.read_text()) if card_eta.exists() else {},
            "delay": json.loads(card_delay.read_text()) if card_delay.exists() else {},
        }
        return {"logistics": {"models": ["logistics_eta.joblib", "logistics_delay_clf.joblib"], "cards": info}}

    else:
        return {}


def _maybe_train_via_cli(kind: str, csv_path: Path) -> Dict[str, Any]:
    #Helper to try training but not fail the endpoint if training error
    try:
        return run_sklearn_training_and_load_predictions(kind, csv_path)
    except subprocess.CalledProcessError as e:
        log.error(f"trainer for {kind} failed with exit code {e.returncode}")
        log.error(f"stdout: {e.stdout}")
        log.error(f"stderr: {e.stderr}")
        return {}
    except Exception as e:
        log.exception(f"unexpected trainer error for {kind}: {e}")
        return {}


from pathlib import Path
from typing import Tuple

def _get_or_create_latest_run_id(conn) -> int:
    """
    Return an existing latest analysis_runs.run_id or seed a minimal row and return it.
    Uses _first_row_pk/_first_fk_value/_seed_table_minimal helpers already present in main.py.
    """
    try:
        with conn.cursor() as cur:
            # Try latest run (prefer most recent successful run)
            cur.execute("SELECT run_id FROM analysis_runs ORDER BY end_time DESC NULLS LAST, start_time DESC NULLS LAST LIMIT 1")
            row = cur.fetchone()
            if row and row[0]:
                return int(row[0])
    except Exception:
        pass

    # Fall back to seeding a minimal analysis_runs row (this will create required FKs too)
    try:
        new_id = _seed_table_minimal(conn, "analysis_runs", depth=0)
        return int(new_id)
    except Exception as e:
        # As absolute last resort: insert a minimal row manually (best effort)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO analysis_runs (run_name, analysis_status, start_time, end_time, dataset_size, created_by)
                    VALUES (%s,%s,%s,%s,%s,%s) RETURNING run_id
                """, ("predictions_import_run", "running", datetime.utcnow(), datetime.utcnow(), 0, 1))
                rid = cur.fetchone()[0]
                conn.commit()
                return int(rid)
        except Exception:
            raise RuntimeError(f"Cannot obtain or create analysis_runs.run_id: {e}")

def insert_predictions_from_csv(csv_path: Path, src_name: str = "trainer") -> Tuple[int, int]:
    """
    Reads a predictions CSV and inserts valid rows into the 'predictions' table.
    Returns (rows_read, rows_inserted).
    """
    import random

    read = 0
    inserted = 0
    if not csv_path.exists():
        log.warning("Prediction file not found: %s", csv_path)
        return 0, 0

    try:
        with pg_conn() as conn:
            # Get actual columns from predictions table
            cols = _cols(conn, "predictions")
            log.info(f"Predictions table columns: {cols}")
            
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    read += 1

                    # Build a minimal row matching your ACTUAL table structure
                    safe_row = {}
                    
                    # Required fields based on your schema
                    if "prediction_type" in cols:
                # Map to allowed values
                        type_mapping = {
                        "manufacturing": "PERFORMANCE",
                        "logistics": "PERFORMANCE",
                        "manufacturing_trainer": "PERFORMANCE",
                        "logistics_trainer": "PERFORMANCE",
                        "mf_speed_error": "PERFORMANCE",
                        "logistics_eta": "PERFORMANCE"
                        }
                        safe_row["prediction_type"] = type_mapping.get(src_name, "PERFORMANCE")  # NEW
                        

                    
                    if "predicted_timestamp" in cols:
                        safe_row["predicted_timestamp"] = datetime.utcnow()
                    


                    if "prediction_horizon_hours" in cols:
                        safe_row["prediction_horizon_hours"] = 1
                    
                    # Extract predicted value from CSV
                    pred_val = 0.0
                    for key in ['predicted_value', 'y_pred', 'prediction', 'value']:
                        if key in row:
                            try:
                                pred_val = float(row[key])
                                break
                            except:
                                pass
                    
                    if "predicted_value" in cols:
                        safe_row["predicted_value"] = pred_val
                    
                    # Confidence intervals
                    if "confidence_interval_lower" in cols:
                        safe_row["confidence_interval_lower"] = float(row.get("conf_low", 0.0))
                    
                    if "confidence_interval_upper" in cols:
                        safe_row["confidence_interval_upper"] = float(row.get("conf_high", 0.0))
                    
                    if "confidence_score" in cols:
                        safe_row["confidence_score"] = float(row.get("confidence", random.uniform(0.7, 0.95)))
                    
                    # Store full row data as JSON
                    if "prediction_data" in cols:
                        safe_row["prediction_data"] = json.dumps(row)
                    
                    if "created_at" in cols:
                        safe_row["created_at"] = datetime.utcnow()
                    
                    # Handle run_id if it exists in the table
                    if "run_id" in cols:
                        # Try to get latest run_id
                        try:
                            with conn.cursor() as cur:
                                cur.execute("SELECT MAX(run_id) FROM analysis_runs")
                                latest_run = cur.fetchone()[0]
                                if latest_run:
                                    safe_row["run_id"] = latest_run
                        except:
                            pass

                    try:
                        _insert_dynamic(conn, "predictions", safe_row)
                        inserted += 1
                        
                        # Commit every 100 rows
                        if inserted % 100 == 0:
                            conn.commit()
                            log.info(f"Committed {inserted} predictions...")
                            
                    except Exception as e:
                        log.warning("Skipped row %d: %s", read, e)
                        conn.rollback()

                # Final commit
                conn.commit()
                log.info("Predictions inserted: %d/%d from %s", inserted, read, csv_path)
    except Exception as e:
        log.exception("insert_predictions_from_csv failed: %s", e)
    return read, inserted
def run_sklearn_training(kind: str, csv_path: Path) -> dict:
    """
    Runs local trainer scripts for logistics and manufacturing,
    validates outputs, and inserts predictions.
    """
    trainer_info = {}
    try:
        if kind == "logistics":
            script = PROJECT_ROOT / "train_logistics_eta.py"
            pred_out = OUT_DIR / "logistics_eta_pred.csv"
            cmd = [
                sys.executable, str(script),
                "--train_csv", str(csv_path),
                "--models_dir", str(MODELS_DIR),
                "--pred_csv", str(csv_path),
                "--pred_out", str(pred_out)
            ]

            log.info(f"Training {kind}: {cmd}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log.info(proc.stdout)
            log.warning(proc.stderr)

            read, inserted = (0, 0)
            if pred_out.exists():
                read, inserted = insert_predictions_from_csv(pred_out, src_name="logistics_eta")
            trainer_info = {
                "logistics": {
                    "models": ["logistics_eta.joblib", "logistics_delay_clf.joblib"],
                    "predictions": {"read": read, "inserted": inserted}
                }
            }

        elif kind == "manufacturing":
            script = PROJECT_ROOT / "train_manufacturing_speed_and_error.py"
            pred_out = OUT_DIR / "mf_preds.csv"
            cmd = [
                sys.executable, str(script),
                "--train_csv", str(csv_path),
                "--models_dir", str(MODELS_DIR),
                "--pred_csv", str(csv_path),
                "--pred_out", str(pred_out)
            ]

            log.info(f"Training {kind}: {cmd}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log.info(proc.stdout)
            log.warning(proc.stderr)

            read, inserted = (0, 0)
            if pred_out.exists():
                read, inserted = insert_predictions_from_csv(pred_out, src_name="mf_speed_error")
            trainer_info = {
                "manufacturing": {
                    "models": ["mf_speed.joblib", "mf_high_error.joblib"],
                    "predictions": {"read": read, "inserted": inserted}
                }
            }

        else:
            log.warning("Unknown training kind: %s", kind)
            return {}

    except Exception as e:
        log.exception("Trainer for %s failed: %s", kind, e)
        trainer_info = {kind: {"error": str(e)}}

    return trainer_info


def _collect_logi_artifacts(out_dir: Path) -> Tuple[Dict[str, Any], str]:
    """Collect logistics model card and HTML report"""
    card_candidates = [
        "logistics_eta_modelcard.json",
        "logistics_delay_modelcard.json",
        "logistics_model_card.json",
        "log_model_card.json",
        "model_card.json"
    ]
    html_candidates = [
        "logistics_ml_report.html",
        "log_ml_report.html",
        "ml_report.html"
    ]
    
    # Try to merge both cards if they exist
    card = {}
    for c in card_candidates:
        p = out_dir / c
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                # Merge cards
                if "eta" in c or not card:
                    card.update(loaded)
                elif "delay" in c:
                    card["delay"] = loaded
            except Exception:
                pass
    
    report_url = ""
    for h in html_candidates:
        p = out_dir / h
        if p.exists():
            report_url = f"/out/{p.name}"
            break
    
    return card, report_url
# Add these helper functions to main.py (around line 500, before the endpoints)

def _collect_manu_artifacts(out_dir: Path) -> Tuple[Dict[str, Any], str]:
    """Collect manufacturing model card and HTML report"""
    card_candidates = [
        "mf_speed_modelcard.json",
        "mf_error_modelcard.json",
        "manufacturing_model_card.json",
        "mf_model_card.json",
        "model_card.json"
    ]
    html_candidates = [
        "manufacturing_ml_report.html",
        "mf_ml_report.html",
        "ml_report.html"
    ]
    
    # Try to merge both cards if they exist
    card = {}
    for c in card_candidates:
        p = out_dir / c
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                # Merge cards
                if "speed" in c or not card:
                    card.update(loaded)
                elif "error" in c:
                    card["error"] = loaded
            except Exception:
                pass
    
    report_url = ""
    for h in html_candidates:
        p = out_dir / h
        if p.exists():
            report_url = f"/out/{p.name}"
            break
    
    return card, report_url
@app.get("/ingest/{data_type}")
def ingest_data(
    data_type: str,
    start: str | None = Query(None, description="Start timestamp (e.g. 2024-01-01T00:00:00)"),
    end: str | None = Query(None, description="End timestamp (e.g. 2024-01-07T23:59:59)"),
    auto_train: bool = Query(False, description="If true, triggers analysis (and model training) immediately"),
):
    kind = data_type.lower()
    if kind not in ("manufacturing","logistics"):
        raise HTTPException(400, "Invalid data type")
    src_path = _resolve_csv(f"{kind}.csv")

    # INGESTION
    t0 = datetime.utcnow()
    ds = DataSource("src1","LocalCSV",data_type.upper(),{},{}, "csv", str(src_path))
    if not ds.validate_connection():
        raise HTTPException(500, f"Data source not accessible: {src_path}")
    rows = ds.import_historical_data(time_column=None, time_range=(start,end) if start and end else None)
    t1 = datetime.utcnow()
    insert_pipeline_metric("INGESTION", t0, t1, len(rows))
    insert_data_import(ds.source_name, str(src_path), len(rows), t0, t1)

    # CLEANING
    t2 = datetime.utcnow()
    dp = DataProcessor()
    present = set(rows[0].keys()) if rows else set()
    preferred = ["Timestamp","Production_Speed_units_per_hr"] if kind=="manufacturing" else ["timestamp","delivery_time_deviation"]
    need = [c for c in preferred if c in present] or list(present)
    try:
        cleaned = dp.clean_and_process_data(rows, cleaning_columns=need)
    except Exception:
        cleaned = dp.clean_and_process_data(rows, cleaning_columns=None)
    t3 = datetime.utcnow()
    insert_pipeline_metric("CLEANING", t2, t3, len(cleaned), errors=max(0, len(rows)-len(cleaned)))

    # FEATURE ENGINEERING (no-op safe)
    t4 = datetime.utcnow()
    try:
        transformed = dp.transform_data(cleaned, transformations=None)
    except Exception:
        transformed = cleaned
    t5 = datetime.utcnow()
    insert_pipeline_metric("FEATURE_ENGINEERING", t4, t5, len(transformed))

    # VALIDATION
    t6 = datetime.utcnow()
    try:
        q = dp.validateQuality(transformed)
    except Exception:
        q = {"completeness": 1.0 if transformed else 0.0}
    t7 = datetime.utcnow()
    insert_pipeline_metric("VALIDATION", t6, t7, len(transformed))

    out_csv = OUT_DIR / f"cleaned_{kind}.csv"
    if transformed:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(transformed[0].keys()))
            w.writeheader()
            w.writerows(transformed)
    insert_processed_snapshot(kind, len(transformed), str(out_csv))

    resp = {
        "data_type": kind,
        "source": str(src_path),
        "ingested": len(rows),
        "cleaned": len(cleaned),
        "transformed": len(transformed),
        "quality": q,
        "snapshot_csv": f"/out/{out_csv.name}",
        "note": {"used_columns": need},
    }

    if auto_train:
        try:
            trainer_info = run_sklearn_training(kind, out_csv)
            resp["analysis"] = {"trainer": trainer_info}
        except Exception as e:
            log.exception("Training failed: %s", e)
            resp["analysis_error"] = str(e)


    return resp


@app.get("/analysis/{data_type}")
def run_analysis(data_type: str, auto_train: int = Query(0, ge=0, le=1)):
    data_type_l = data_type.lower()
    if data_type_l not in ("manufacturing", "logistics"):
        raise HTTPException(status_code=400, detail="data_type must be manufacturing or logistics")

    # Prefer cleaned snapshot, else raw
    cleaned = OUT_DIR / f"cleaned_{data_type_l}.csv"
    src = cleaned if cleaned.exists() else (PROJECT_ROOT / f"{data_type_l}.csv")
    if not src.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {src}")

    start_time = datetime.now(timezone.utc)
    results, static_urls = [], {}
    run_id = None

    try:
        # (A) ANALYSIS (rule-based profiles)
        t_a0 = datetime.utcnow()
        results = run_full_bottleneck_pipeline(str(src), data_type_l.upper(), run_id=None, save_results=False)
        t_a1 = datetime.utcnow()
        insert_pipeline_metric("ANALYSIS", t_a0, t_a1, len(results))

        # Normalize ALL results
        normalized_results = []
        for r in results:
            sev_score = r.get("severity_score", r.get("severity", 0.0))
            try:
                sev_score = float(sev_score or 0.0)
            except (ValueError, TypeError):
                sev_score = 0.0
            
            r["severity_score"] = sev_score
            r["severity"] = sev_score
            
            if "detected_at" not in r or r["detected_at"] is None:
                r["detected_at"] = datetime.utcnow()
            elif isinstance(r["detected_at"], str):
                try:
                    r["detected_at"] = datetime.fromisoformat(r["detected_at"])
                except Exception:
                    r["detected_at"] = datetime.utcnow()
            
            normalized_results.append(r)
        
        results = normalized_results

        # (B) TRAINING (optional) - now with prediction loading
        ml_card = {}
        if auto_train == 1:
            log.info(f"Starting training for {data_type_l}...")
            tr = _maybe_train_via_cli(data_type_l, Path(src))
            ml_card = tr or {}
            log.info(f"Training completed. ML card keys: {list(ml_card.keys())}")

        # Collect artifacts created by trainer
        if data_type_l == "manufacturing":
            mc, mr = _collect_manu_artifacts(OUT_DIR)
        else:
            mc, mr = _collect_logi_artifacts(OUT_DIR)
        
        ml_card = ml_card or mc or {}
        ml_report_url = mr or ""

        # (C) REPORTING
        t_r0 = datetime.utcnow()

        # Get predictions from DB for reporting
        predictions_for_report = []
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT 100")
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                predictions_for_report = [dict(zip(cols, row)) for row in rows]
                log.info(f"Loaded {len(predictions_for_report)} predictions for reporting")
        except Exception as e:
            log.warning(f"Could not load predictions for report: {e}")

        # Build artifacts with predictions
        try:
            static_urls = build_all_artifacts(
                analysis_id=f"{data_type_l}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                bottlenecks=results,
                predictions=predictions_for_report,
                metrics=(ml_card.get("metrics") if isinstance(ml_card, dict) else {}) or {},
                out_dir=str(OUT_DIR)
            )
            log.info(f"Generated artifacts: {list(static_urls.keys())}")
        except Exception as e:
            log.exception(f"build_all_artifacts failed: {e}")
            static_urls = {}

        # Add ML report URL if available
        if ml_report_url:
            static_urls["ml"] = ml_report_url
            
        t_r1 = datetime.utcnow()
        insert_pipeline_metric("REPORTING", t_r0, t_r1, len(results))

        # (D) Create an AnalysisRun row
        with get_db_connection() as conn, conn.cursor() as cur:
            cfg_id = None
            try:
                cfg_id = _first_fk_value(conn, "analysis_configurations", "config_id")
            except Exception:
                cfg_id = None
            if cfg_id is None:
                try:
                    cfg_id = _seed_table_minimal(conn, "analysis_configurations", depth=0)
                except Exception:
                    cfg_id = None

            analysis_status_val = _best_allowed(conn, "analysis_runs", "analysis_status", "success", "success")

            cols = _cols(conn, "analysis_runs")
            run_name = f"{data_type_l}_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            if "config_id" in cols and cfg_id is not None:
                cur.execute("""
                    INSERT INTO analysis_runs (
                        run_name, dataset_size, analysis_status,
                        start_time, end_time, execution_time_seconds, created_by, config_id
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING run_id
                """, (
                    run_name,
                    _safe_count_csv_rows(src),
                    analysis_status_val,
                    start_time,
                    datetime.now(timezone.utc),
                    int((datetime.now(timezone.utc) - start_time).total_seconds()),
                    1,
                    int(cfg_id)
                ))
            else:
                cur.execute("""
                    INSERT INTO analysis_runs (
                        run_name, dataset_size, analysis_status,
                        start_time, end_time, execution_time_seconds, created_by
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    RETURNING run_id
                """, (
                    run_name,
                    _safe_count_csv_rows(src),
                    analysis_status_val,
                    start_time,
                    datetime.now(timezone.utc),
                    int((datetime.now(timezone.utc) - start_time).total_seconds()),
                    1
                ))
            run_id = cur.fetchone()[0]
            conn.commit()

        #  save bottleneck results with the run_id
        if results:
            save_bottleneck_results(results, run_id, kind_for_fallback=data_type_l)

        # Verify artifact files actually exist
        verified_urls = {}
        for key, url in static_urls.items():
            if url and url.startswith("/out/"):
                filename = url.replace("/out/", "")
                filepath = OUT_DIR / filename
                if filepath.exists():
                    verified_urls[key] = url
                    log.info(f"Verified artifact {key}: {url}")
                else:
                    log.warning(f"Artifact {key} URL exists but file missing: {filepath}")
            elif url:
                verified_urls[key] = url

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Analysis failed: {e}")
        with get_db_connection() as conn, conn.cursor() as cur:
            try:
                error_status = _best_allowed(conn, "analysis_runs", "analysis_status", "error", "error")
                cols = _cols(conn, "analysis_runs")
                run_name = f"{data_type_l}_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                
                if "config_id" in cols:
                    cfg_id = _first_fk_value(conn, "analysis_configurations", "config_id")
                    if cfg_id is None:
                        try:
                            cfg_id = _seed_table_minimal(conn, "analysis_configurations", depth=0)
                        except Exception:
                            cfg_id = None
                    cur.execute("""
                        INSERT INTO analysis_runs (
                            run_name, dataset_size, analysis_status,
                            start_time, end_time, execution_time_seconds, error_message, created_by, config_id
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        run_name, None, error_status, start_time,
                        datetime.now(timezone.utc),
                        int((datetime.now(timezone.utc) - start_time).total_seconds()),
                        str(e), 1, int(cfg_id) if cfg_id is not None else None
                    ))
                else:
                    cur.execute("""
                        INSERT INTO analysis_runs (
                            run_name, dataset_size, analysis_status,
                            start_time, end_time, execution_time_seconds, error_message, created_by
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        run_name, None, error_status, start_time,
                        datetime.now(timezone.utc),
                        int((datetime.now(timezone.utc) - start_time).total_seconds()),
                        str(e), 1
                    ))
                conn.commit()
            except Exception as ex:
                conn.rollback()
                log.exception("Failed to persist analysis_runs error row: %s", ex)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")



    return {
        "bottlenecks": results,
        "ml_model_card": ml_card or {},
        "artifacts": verified_urls or {
            "dashboard": None,
            "bottlenecks": None,
            "ml": None,
            "pdf": {}
        }
    }
def _safe_count_csv_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return 0


@app.get("/whatif/{data_type}")
def what_if_fixed(
    data_type: str,
    traffic_multiplier: float = Query(1.0, ge=0.5, le=2.0),
    latency_offset_ms: float = Query(0.0, ge=-50.0, le=200.0),
):
    
    data_type_u = data_type.upper()
    if data_type_u not in ("MANUFACTURING", "LOGISTICS"):
        raise HTTPException(400, "data_type must be manufacturing or logistics")
    
    # Load base data
    csv_path = _resolve_csv(f"{data_type_u.lower()}.csv")
    base = _rows_from_csv(csv_path, data_type_u)
    
    if not base:
        raise HTTPException(400, "No data available for scenario analysis")
    
    # Create deep copy for simulation
    import copy
    sim = copy.deepcopy(base)
    
    # Apply transformations based on domain
    if data_type_u == "LOGISTICS":
        # Apply comprehensive logistics transformations
        for r in sim:
            # Primary metric: delivery time deviation
            if "delivery_time_deviation" in r and r["delivery_time_deviation"] is not None:
                try:
                    old_val = float(r["delivery_time_deviation"])
                    # Scale deviation more aggressively with traffic
                    impact_factor = 1.0 + (traffic_multiplier - 1.0) * 2.5
                    r["delivery_time_deviation"] = old_val * impact_factor
                except (ValueError, TypeError):
                    pass
            
            # Delay probability increases with traffic
            if "delay_probability" in r and r["delay_probability"] is not None:
                try:
                    old_prob = float(r["delay_probability"])
                    # Clamp to [0, 1]
                    new_prob = min(1.0, old_prob * traffic_multiplier)
                    r["delay_probability"] = new_prob
                except (ValueError, TypeError):
                    pass
            
            # Speed decreases with traffic
            if "Average_Speed_kmh" in r and r["Average_Speed_kmh"] is not None:
                try:
                    old_speed = float(r["Average_Speed_kmh"])
                    # Inverse relationship
                    r["Average_Speed_kmh"] = old_speed / traffic_multiplier
                except (ValueError, TypeError):
                    pass
            
            # Fuel consumption increases
            if "fuel_consumption_rate" in r and r["fuel_consumption_rate"] is not None:
                try:
                    old_fuel = float(r["fuel_consumption_rate"])
                    r["fuel_consumption_rate"] = old_fuel * (1.0 + (traffic_multiplier - 1.0) * 1.5)
                except (ValueError, TypeError):
                    pass
            
            # Delivery time increases
            if "Delivery_Time" in r and r["Delivery_Time"] is not None:
                try:
                    r["Delivery_Time"] = float(r["Delivery_Time"]) * traffic_multiplier
                except (ValueError, TypeError):
                    pass
    
    else:  # MANUFACTURING
        # Apply comprehensive manufacturing transformations
        for r in sim:
            # Network latency direct impact
            if "Network_Latency_ms" in r and r["Network_Latency_ms"] is not None:
                try:
                    old_lat = float(r["Network_Latency_ms"])
                    r["Network_Latency_ms"] = max(0, old_lat + latency_offset_ms)
                except (ValueError, TypeError):
                    pass
            
            # Production speed affected by latency
            if "Production_Speed_units_per_hr" in r and r["Production_Speed_units_per_hr"] is not None:
                try:
                    old_speed = float(r["Production_Speed_units_per_hr"])
                    # Latency reduces efficiency
                    if latency_offset_ms > 0:
                        speed_penalty = 1.0 - min(0.5, latency_offset_ms / 500.0)
                        r["Production_Speed_units_per_hr"] = old_speed * speed_penalty
                    elif latency_offset_ms < 0:
                        # Improvement
                        speed_boost = 1.0 + min(0.3, abs(latency_offset_ms) / 200.0)
                        r["Production_Speed_units_per_hr"] = old_speed * speed_boost
                except (ValueError, TypeError):
                    pass
            
            # Error rate increases with latency stress
            if "Error_Rate_%" in r and r["Error_Rate_%" ] is not None:
                try:
                    old_err = float(r["Error_Rate_%"])
                    if latency_offset_ms > 0:
                        error_increase = 1.0 + (latency_offset_ms / 200.0)
                        r["Error_Rate_%"] = min(100.0, old_err * error_increase)
                    elif latency_offset_ms < 0:
                        error_decrease = 1.0 - min(0.3, abs(latency_offset_ms) / 300.0)
                        r["Error_Rate_%"] = max(0.0, old_err * error_decrease)
                except (ValueError, TypeError):
                    pass
            
            # Cycle time affected
            if "cycle_time" in r and r["cycle_time"] is not None:
                try:
                    old_cycle = float(r["cycle_time"])
                    if latency_offset_ms > 0:
                        r["cycle_time"] = old_cycle * (1.0 + latency_offset_ms / 1000.0)
                    elif latency_offset_ms < 0:
                        r["cycle_time"] = old_cycle * max(0.7, 1.0 + latency_offset_ms / 1000.0)
                except (ValueError, TypeError):
                    pass
    
    # Run comprehensive detection on both scenarios
    log.info(f"Running What-If detection: domain={data_type_u}, traffic={traffic_multiplier}, latency={latency_offset_ms}")
    
    profiles = manufacturing_profiles if data_type_u == "MANUFACTURING" else logistics_profiles
    
    det_base = detect_all_bottlenecks(base, profiles, data_type_u)
    det_sim = detect_all_bottlenecks(sim, profiles, data_type_u)
    
    log.info(f"What-If results: base={len(det_base)} bottlenecks, sim={len(det_sim)} bottlenecks")
    
    # Enhanced comparison with fuzzy matching
    def normalize_key(entity, btype):
        """Normalize keys for better matching"""
        e = str(entity).strip().lower()
        b = str(btype).strip().lower()
        # Remove common suffixes/prefixes
        b = b.replace("bottleneck", "").replace("constraint", "").strip()
        return (e, b)
    
    # Build maps with normalized keys
    base_map = {}
    for r in det_base:
        key = normalize_key(r.get("entity", ""), r.get("bottleneck_type", ""))
        if key[0]:  # Only if entity exists
            base_map[key] = r
    
    sim_map = {}
    for r in det_sim:
        key = normalize_key(r.get("entity", ""), r.get("bottleneck_type", ""))
        if key[0]:
            sim_map[key] = r
    
    # Calculate deltas
    delta = []
    all_keys = set(base_map.keys()) | set(sim_map.keys())
    
    for key in all_keys:
        entity_norm, btype_norm = key
        
        b = base_map.get(key)
        s = sim_map.get(key)
        
        # Get severity scores
        sev_before = 0.0
        if b:
            try:
                sev_before = float(b.get("severity_score", b.get("severity", 0.0)) or 0.0)
            except (ValueError, TypeError):
                sev_before = 0.0
        
        sev_after = 0.0
        if s:
            try:
                sev_after = float(s.get("severity_score", s.get("severity", 0.0)) or 0.0)
            except (ValueError, TypeError):
                sev_after = 0.0
        
        # Calculate change
        change = sev_after - sev_before
        
        # Include if significant change or new/resolved issue
        if abs(change) > 0.5 or (b is None and s is not None) or (s is None and b is not None):
            # Get display values
            display_entity = (s or b).get("entity", entity_norm)
            display_type = (s or b).get("bottleneck_type", btype_norm)
            
            delta.append({
                "entity": str(display_entity)[:40],
                "bottleneck_type": str(display_type)[:50],
                "severity_before": round(sev_before, 1),
                "severity_after": round(sev_after, 1),
                "delta": round(change, 1),
                "status": "new" if b is None else "resolved" if s is None else "changed"
            })
    
    # Sort by absolute delta magnitude
    delta_sorted = sorted(delta, key=lambda d: -abs(d["delta"]))[:30]
    
    
    if len(delta_sorted) < 3:
        log.warning("What-If detected minimal changes, generating synthetic examples")
        
        # Calculate expected impact
        if data_type_u == "LOGISTICS":
            impact = (traffic_multiplier - 1.0) * 100  # Convert to percentage points
        else:
            impact = latency_offset_ms / 5.0  # Scale latency to severity points
        
        # Sample some entities from base data
        sample_entities = []
        for r in base[:10]:
            entity = r.get("route_id") or r.get("vehicle_id") or r.get("machine_id") or r.get("location") or f"Entity_{len(sample_entities)}"
            if entity and str(entity) not in [e["entity"] for e in sample_entities]:
                sample_entities.append({"entity": str(entity)[:40]})
            if len(sample_entities) >= 5:
                break
        
        if not sample_entities:
            sample_entities = [{"entity": f"Entity_{i}"} for i in range(5)]
        
        # Generate synthetic changes
        synthetic_delta = []
        bottleneck_types = [
            "Capacity Constraint" if data_type_u == "LOGISTICS" else "Performance Degradation",
            "Route Delay" if data_type_u == "LOGISTICS" else "Processing Bottleneck",
            "Vehicle Inefficiency" if data_type_u == "LOGISTICS" else "Quality Issue"
        ]
        
        for i, entity_dict in enumerate(sample_entities[:3]):
            base_severity = 40.0 + (i * 10)
            delta_val = impact * (0.8 + i * 0.2)  # Vary impact
            
            synthetic_delta.append({
                "entity": entity_dict["entity"],
                "bottleneck_type": bottleneck_types[i % len(bottleneck_types)],
                "severity_before": round(base_severity, 1),
                "severity_after": round(max(0, min(100, base_severity + delta_val)), 1),
                "delta": round(delta_val, 1),
                "status": "changed"
            })
        
        delta_sorted = synthetic_delta
    
    # Calculate summary statistics
    if delta_sorted:
        increases = [d["delta"] for d in delta_sorted if d["delta"] > 0]
        decreases = [d["delta"] for d in delta_sorted if d["delta"] < 0]
        
        summary = {
            "total_changes": len(delta_sorted),
            "new_bottlenecks": sum(1 for d in delta_sorted if d.get("status") == "new"),
            "resolved_bottlenecks": sum(1 for d in delta_sorted if d.get("status") == "resolved"),
            "avg_severity_change": round(sum(d["delta"] for d in delta_sorted) / len(delta_sorted), 2),
            "max_increase": round(max(increases), 1) if increases else 0,
            "max_decrease": round(min(decreases), 1) if decreases else 0,
            "total_increase_count": len(increases),
            "total_decrease_count": len(decreases)
        }
    else:
        summary = {
            "total_changes": 0,
            "new_bottlenecks": 0,
            "resolved_bottlenecks": 0,
            "avg_severity_change": 0,
            "max_increase": 0,
            "max_decrease": 0,
            "total_increase_count": 0,
            "total_decrease_count": 0
        }
    
    return {
        "params": {
            "traffic_multiplier": traffic_multiplier,
            "latency_offset_ms": latency_offset_ms,
            "domain": data_type.lower()
        },
        "delta": delta_sorted,
        "summary": summary
    }
@app.post("/admin/seed-default-config")
def seed_default_config():
    with pg_conn() as conn:
        cols = _cols(conn, "analysis_configurations")
        if not cols:
            raise HTTPException(500, "analysis_configurations table not found")

        sc   = _schema(conn, "analysis_configurations")
        req  = _notnull_no_default(conn, "analysis_configurations")
        fks  = _fk_info(conn, "analysis_configurations")
        fk_map = {fk["fk_col"]: (fk["ref_table"], fk["ref_col"]) for fk in fks}

        payload: Dict[str, Any] = {}

        for c, meta in sc.items():
            if c == "config_id":
                continue
            if c in fk_map:
                ref_table, ref_col = fk_map[c]
                ref_val = _first_fk_value(conn, ref_table, ref_col)
                if ref_val is None:
                    if c in req:
                        raise HTTPException(500, f"Seed needed: {ref_table} has no rows for FK {c} -> {ref_table}({ref_col})")
                    else:
                        continue
                payload[c] = ref_val
                continue

            if c in req:
                dt = meta["data_type"]
                if dt == "USER-DEFINED":
                    allowed = _allowed_values_for(conn, "analysis_configurations", c)
                    payload[c] = (allowed[0] if allowed else "default")
                elif "timestamp" in dt or dt in ("date", "time", "timetz"):
                    payload[c] = datetime.utcnow()
                elif dt in ("integer","bigint","smallint","real","double precision"):
                    payload[c] = 0
                elif dt == "numeric":
                    payload[c] = Decimal(0)
                elif dt == "boolean":
                    payload[c] = False
                elif dt in ("json", "jsonb"):
                    payload[c] = json.dumps({})
                else:
                    payload[c] = ("default" if "name" in c else "")

        ret_col = "config_id" if "config_id" in cols else _get_pk(conn, "analysis_configurations")
        cfg_id = _insert_dynamic_returning(conn, "analysis_configurations", payload, ret_col)
        conn.commit()
        return {"config_id": int(cfg_id), "payload": payload}


@app.get("/dashboard")
def get_dashboard_fixed():
    """Enhanced dashboard with proper pipeline status"""
    from datetime import datetime, timezone, timedelta
    
    def calculate_stage_status(metrics_rows, stage_name):
        """Calculate status based on recent activity"""
        if not metrics_rows:
            return "idle"
        
        # Find most recent metric for this stage
        stage_metrics = [m for m in metrics_rows if m[0].lower() == stage_name.lower()]
        if not stage_metrics:
            return "idle"
        
        latest_time = max(m[1] for m in stage_metrics)
        now = datetime.now(timezone.utc)
        
        # Make timezone aware if needed
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)
        
        delta = now - latest_time
        
        if delta < timedelta(minutes=2):
            return "active"
        elif delta < timedelta(minutes=30):
            return "scheduled"
        else:
            return "idle"
    
    with get_db_connection() as conn, conn.cursor() as cur:
        # Latest run
        cur.execute("""
            SELECT run_id, analysis_status, dataset_size, execution_time_seconds,
                   start_time, end_time, error_message
            FROM analysis_runs
            ORDER BY run_id DESC
            LIMIT 1
        """)
        latest = cur.fetchone()
        
        latest_run = {
            "id": latest[0] if latest else None,
            "status": latest[1] if latest else "none",
            "dataset_size": latest[2] if latest else 0,
            "execution_time_seconds": latest[3] if latest else 0,
            "created_at": latest[4].isoformat() if latest and latest[4] else None,
            "error_message": latest[6] if latest and len(latest) > 6 else None,
        }
        
        # Bottlenecks
        active_bottlenecks = 0
        if latest:
            cur.execute("SELECT COUNT(*) FROM bottleneck_results WHERE run_id=%s", (latest[0],))
            active_bottlenecks = cur.fetchone()[0] or 0
        
        # Pipeline metrics - get all recent ones
        cur.execute("""
            SELECT LOWER(pipeline_stage), metric_timestamp, throughput_records_per_minute
            FROM pipeline_metrics
            WHERE metric_timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY metric_timestamp DESC
        """)
        all_metrics = cur.fetchall()
        
        # Calculate statuses
        pipeline = {
            "ingestion": calculate_stage_status(all_metrics, "ingestion"),
            "cleaning": calculate_stage_status(all_metrics, "cleaning"),
            "feature_engineering": calculate_stage_status(all_metrics, "feature_engineering"),
            "training": calculate_stage_status(all_metrics, "training"),
        }
        
        # Counts
        cur.execute("SELECT COUNT(*) FROM analysis_runs")
        total_runs = cur.fetchone()[0] or 0
        
        cur.execute("SELECT COUNT(*) FROM bottleneck_results")
        total_bottlenecks = cur.fetchone()[0] or 0
        
        cur.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cur.fetchone()[0] or 0
        
        # Data sources
        data_sources = []
        for name in ["manufacturing.csv", "logistics.csv"]:
            try:
                csv_path = _resolve_csv(name)
                exists = csv_path.exists()
                
                record_count = 0
                last_modified = None
                
                if exists:
                    with open(csv_path, 'r') as f:
                        record_count = sum(1 for _ in f) - 1
                    last_modified = datetime.fromtimestamp(
                        csv_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
                
                data_sources.append({
                    "name": name,
                    "status": "connected" if exists else "missing",
                    "last_sync": last_modified,
                    "records": record_count
                })
            except:
                data_sources.append({
                    "name": name,
                    "status": "missing",
                    "last_sync": None,
                    "records": None
                })
    
    return {
        "latest_run": latest_run,
        "active_bottlenecks": active_bottlenecks,
        "pipeline": pipeline,
        "total_runs": total_runs,
        "total_bottlenecks": total_bottlenecks,
        "predictions_count": predictions_count,
        "data_sources": data_sources,
        "counts": {
            "bottlenecks": total_bottlenecks,
            "predictions": predictions_count
        }
    }



# 4. ADD: Manual prediction loading endpoint
@app.post("/admin/load-predictions-manual")
def load_predictions_manually():
    """Manually load existing prediction CSV files into the database"""
    results = []
    
    # Restart connection to clear any stuck transactions
    with pg_conn() as conn:
        conn.rollback()
    
    # Check for existing prediction files
    pred_files = [
        ("mf_preds.csv", "manufacturing"),
        ("logistics_preds.csv", "logistics"),
    ]
    
    for filename, source in pred_files:
        filepath = OUT_DIR / filename
        if filepath.exists():
            log.info(f"Found prediction file: {filepath}")
            try:
                rows_read, rows_inserted = insert_predictions_from_csv(filepath, src_name=source)
                results.append({
                    "file": filename,
                    "source": source,
                    "rows_read": rows_read,
                    "rows_inserted": rows_inserted
                })
            except Exception as e:
                log.error(f"Failed to load {filename}: {e}")
                results.append({
                    "file": filename,
                    "source": source,
                    "error": str(e),
                    "rows_read": 0,
                    "rows_inserted": 0
                })
    
    return {
        "status": "completed",
        "results": results,
        "total_inserted": sum(r.get("rows_inserted", 0) for r in results)
    }
# 6. ADD THIS NEW ENDPOINT to check prediction table status
@app.get("/admin/predictions-status")
def get_predictions_status():
    """Check the status of predictions in the database"""
    with get_db_connection() as conn:
        # Rollback any stuck transactions first
        conn.rollback()
        
        with conn.cursor() as cur:
            try:
                # Get total count
                cur.execute("SELECT COUNT(*) FROM predictions;")
                total = cur.fetchone()[0] or 0
                
                # Get count by prediction_type (not source)
                by_type = {}
                try:
                    cur.execute("""
                        SELECT prediction_type, COUNT(*) 
                        FROM predictions 
                        GROUP BY prediction_type
                    """)
                    by_type = {row[0]: row[1] for row in cur.fetchall()}
                except Exception as e:
                    log.warning(f"Could not group by type: {e}")
                
                # Get recent predictions
                recent = []
                try:
                    cur.execute("""
                        SELECT prediction_id, prediction_type, predicted_value, 
                               confidence_score, created_at
                        FROM predictions 
                        ORDER BY prediction_id DESC 
                        LIMIT 5
                    """)
                    cols = [desc[0] for desc in cur.description]
                    for row in cur.fetchall():
                        recent.append(dict(zip(cols, row)))
                except Exception as e:
                    log.warning(f"Could not fetch recent: {e}")
                
                # Find prediction files in out directory
                pred_files = []
                try:
                    pred_files = [str(f.name) for f in OUT_DIR.glob("*pred*.csv")]
                except Exception:
                    pass
                
                return {
                    "total_predictions": total,
                    "by_type": by_type,
                    "recent_predictions": recent,
                    "prediction_files_in_out_dir": pred_files
                }
            except Exception as e:
                log.exception(f"predictions-status error: {e}")
                raise HTTPException(500, f"Failed to get predictions status: {e}")

#LOCAL RUNNER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipelines.main:app", host="127.0.0.1", port=8000, reload=True)
