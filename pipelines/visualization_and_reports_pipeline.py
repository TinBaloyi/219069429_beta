# visualization_and_reports_pipeline.py - COMPLETE FIXED VERSION
from __future__ import annotations
import os
import json
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

log = logging.getLogger(__name__)

# ---------- Core data classes ----------
@dataclass
class InteractiveVisualization:
    viz_id: str
    viz_type: str
    data_source: Any
    html_fragment: str

@dataclass
class AnalysisReport:
    report_id: str
    analysis_id: str
    content: str          # full HTML (or Markdown -> HTML)
    format: str = "html"

@dataclass
class Dashboard:
    dashboard_id: str
    name: str
    components: List[InteractiveVisualization] = field(default_factory=list)

# ---------- Visualization Engine ----------
class VisualizationEngine:
    """
    Creates interactive Plotly visualizations from analysis outputs.
    Expects dicts/lists/DFS returned by your analysis step (no hardcoding).
    """
    def __init__(self, out_dir: str = "out"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # Helpers
    def _to_df(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def createVisualization(self, *, df: pd.DataFrame, x: str, y: str, kind: str="bar", title: str="") -> InteractiveVisualization:
        vid = str(uuid.uuid4())
        if kind == "bar":
            fig = px.bar(df, x=x, y=y, title=title)
        elif kind == "line":
            fig = px.line(df, x=x, y=y, title=title)
        elif kind == "pie":
            fig = px.pie(df, names=x, values=y, title=title, hole=0)
        else:
            raise ValueError(f"Unsupported viz kind: {kind}")
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return InteractiveVisualization(viz_id=vid, viz_type=kind, data_source=None, html_fragment=html)

    def generateHeatmap(self, *, df: pd.DataFrame, x: str, y: str, z: str, title: str="Heatmap") -> InteractiveVisualization:
        vid = str(uuid.uuid4())
        pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, coloraxis="coloraxis"))
        fig.update_layout(title=title, coloraxis={"colorscale": "Blues"})
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return InteractiveVisualization(viz_id=vid, viz_type="heatmap", data_source=None, html_fragment=html)

    def createTimeline(self, *, df: pd.DataFrame, t_col: str, y_col: str, title: str="Timeline") -> InteractiveVisualization:
        vid = str(uuid.uuid4())
        df2 = df.copy()
        df2[t_col] = pd.to_datetime(df2[t_col], errors="coerce")
        fig = px.line(df2.sort_values(t_col), x=t_col, y=y_col, title=title)
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return InteractiveVisualization(viz_id=vid, viz_type="timeline", data_source=None, html_fragment=html)

    # High-level shortcuts used by reports/dashboards
    def viz_bottleneck_severity(self, bottlenecks: Sequence[dict]) -> InteractiveVisualization:
        df = self._to_df([{
            "entity": b.get("entity") or b.get("group") or b.get("id"),
            "severity": float(b.get("severity_score", 0.0)),
        } for b in bottlenecks])
        df = df.sort_values("severity", ascending=False).head(20)
        return self.createVisualization(df=df, x="entity", y="severity",
                                        kind="bar", title="Bottleneck Severity (Top 20)")

    def viz_bottleneck_distribution(self, bottlenecks: Sequence[dict]) -> InteractiveVisualization:
        df = self._to_df(bottlenecks)
        dist = df.groupby(df["bottleneck_type"].fillna("Unknown")).size().reset_index(name="count")
        return self.createVisualization(df=dist, x="bottleneck_type", y="count",
                                        kind="pie", title="Bottleneck Types Distribution")

    def viz_timeline(self, bottlenecks: Sequence[dict]) -> InteractiveVisualization:
        df = self._to_df(bottlenecks)
        if "detected_at" not in df.columns:
            return self.createVisualization(df=pd.DataFrame({"note":["No timestamps"],"value":[0]}),
                                            x="note", y="value", kind="bar", title="Timeline")
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(0.0)
        return self.createTimeline(df=df, t_col="detected_at", y_col="severity_score",
                                   title="Recent Detections Timeline")

# ---------- Report Generator ----------
class ReportGenerator:
    """
    Builds HTML reports from visualizations and analysis tables.
    """
    def __init__(self, out_dir: str = "out"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _wrap_html(self, title: str, fragments: List[str], extra: str="") -> str:
        return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body {{ font-family: Inter, system-ui, Arial; background:#f8fafc; margin:0; }}
 .wrap {{ max-width: 1100px; margin: 24px auto; background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:18px; }}
 h1 {{ margin: 8px 0 16px 0; font-size: 22px; }}
 .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px; margin:12px 0; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th,td {{ padding: 8px; border-bottom:1px solid #f3f4f6; text-align:left; }}
 .muted {{ color:#6b7280; font-size:12px; }}
</style>
</head><body>
<div class="wrap">
<h1>{title}</h1>
<div class="muted">Generated by Bottleneck Platform</div>
{extra}
{''.join(f'<div class="card">{frag}</div>' for frag in fragments)}
</div></body></html>"""

    def generate_bottleneck_report(self, *, analysis_id: str, bottlenecks: Sequence[dict], metrics: Dict[str,Any]) -> AnalysisReport:
        ve = VisualizationEngine(self.out_dir)
        v1 = ve.viz_bottleneck_severity(bottlenecks)
        v2 = ve.viz_bottleneck_distribution(bottlenecks)
        v3 = ve.viz_timeline(bottlenecks)

        # summary table
        df = pd.DataFrame(bottlenecks)
        tbl_html = "" if df.empty else df.head(30).to_html(index=False)

        fragments = [
            v1.html_fragment,
            v2.html_fragment,
            v3.html_fragment,
            "<h3>Sample of Detections</h3>" + tbl_html
        ]
        extra = f"<h3>Model Metrics</h3><pre>{json.dumps(metrics or {}, indent=2)}</pre>"
        html = self._wrap_html("Bottleneck Report", fragments, extra)
        rid = str(uuid.uuid4())
        return AnalysisReport(report_id=rid, analysis_id=analysis_id, content=html, format="html")

    def generate_prediction_report(self, *, analysis_id: str, predictions: Sequence[dict]) -> AnalysisReport:
        df = pd.DataFrame(predictions)
        if "pred_score" in df.columns:
            fig = px.histogram(df, x="pred_score", nbins=30, title="Prediction Score Distribution")
            frag = fig.to_html(full_html=False, include_plotlyjs="cdn")
        else:
            frag = "<div>No prediction scores available.</div>"
        tbl = "" if df.empty else df.head(50).to_html(index=False)
        html = self._wrap_html("Prediction Report", [frag, "<h3>Sample Predictions</h3>"+tbl])
        rid = str(uuid.uuid4())
        return AnalysisReport(report_id=rid, analysis_id=analysis_id, content=html, format="html")

    def export_report(self, report: AnalysisReport, filename: str) -> str:
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.content)
        return path

# ---------- Dashboard Manager ----------
class DashboardManager:
    """
    Assembles multiple visualizations into a single interactive dashboard HTML.
    """
    def __init__(self, out_dir: str = "out"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def createDashboard(self, name: str, components: List[InteractiveVisualization]) -> Dashboard:
        return Dashboard(dashboard_id=str(uuid.uuid4()), name=name, components=components)

    def publishDashboard(self, dashboard: Dashboard, filename: str="dashboard.html") -> str:
        grid = "".join(
            f'<div class="card">{c.html_fragment}</div>'
            for c in dashboard.components
        )
        html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/><title>{dashboard.name}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body {{ font-family: Inter, system-ui, Arial; background:#eef2ff; margin:0; }}
 .wrap {{ max-width: 1200px; margin: 24px auto; }}
 .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; margin:12px; }}
 .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
 @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<div class="wrap">
  <h1 style="margin:12px">Interactive Dashboard</h1>
  <div class="grid">{grid}</div>
</div></body></html>
"""
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path


# ---------- ENHANCED ML REPORT GENERATOR ----------
def generate_ml_report_enhanced(analysis_id: str, predictions: list, metrics: dict, out_dir: str = "out") -> str:
    """
    Generate a comprehensive ML report with predictions and metrics.
    This is the FIXED version that creates proper visualizations.
    """
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Prepare data
        pred_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()
        
        # Build visualizations
        figs = []
        
        # 1. Metrics bar chart
        if metrics:
            metric_data = []
            for model, model_metrics in metrics.items():
                if isinstance(model_metrics, dict):
                    for metric_name, metric_val in model_metrics.items():
                        if isinstance(metric_val, (int, float)):
                            metric_data.append({
                                "Model": str(model),
                                "Metric": str(metric_name),
                                "Value": float(metric_val)
                            })
            
            if metric_data:
                metrics_df = pd.DataFrame(metric_data)
                fig1 = px.bar(metrics_df, x="Metric", y="Value", color="Model",
                             title="Model Performance Metrics", barmode="group")
                figs.append(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        
        # 2. Prediction distribution
        if not pred_df.empty:
            if "predicted_value" in pred_df.columns:
                fig2 = px.histogram(pred_df, x="predicted_value", nbins=30,
                                  title="Prediction Value Distribution")
                figs.append(fig2.to_html(full_html=False, include_plotlyjs="cdn"))
            
            # 3. Confidence scores
            if "confidence_score" in pred_df.columns:
                fig3 = px.box(pred_df, y="confidence_score",
                             title="Prediction Confidence Scores")
                figs.append(fig3.to_html(full_html=False, include_plotlyjs="cdn"))
            
            # 4. Time series if timestamp exists
            if "predicted_timestamp" in pred_df.columns:
                pred_df["predicted_timestamp"] = pd.to_datetime(pred_df["predicted_timestamp"], errors="coerce")
                time_df = pred_df.dropna(subset=["predicted_timestamp"]).sort_values("predicted_timestamp")
                if not time_df.empty and "predicted_value" in time_df.columns:
                    fig4 = px.line(time_df, x="predicted_timestamp", y="predicted_value",
                                  title="Predictions Over Time")
                    figs.append(fig4.to_html(full_html=False, include_plotlyjs="cdn"))
        
        # Sample predictions table
        table_html = ""
        if not pred_df.empty:
            display_cols = ["prediction_id", "predicted_value", "confidence_score", 
                           "predicted_timestamp", "prediction_type"]
            available_cols = [c for c in display_cols if c in pred_df.columns]
            if available_cols:
                table_html = pred_df[available_cols].head(50).to_html(index=False, classes="table")
        
        # Metrics JSON
        metrics_json = f"<pre>{json.dumps(metrics, indent=2)}</pre>" if metrics else "<p>No metrics available</p>"
        
        # Calculate stats
        total_predictions = len(predictions)
        num_models = len(metrics) if metrics else 0
        avg_confidence = pred_df['confidence_score'].mean() if not pred_df.empty and 'confidence_score' in pred_df.columns else None
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>ML Model Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body {{ font-family: Inter, system-ui, Arial; background:#f8fafc; margin:0; padding: 20px; }}
 .wrap {{ max-width: 1200px; margin: 0 auto; background:#fff; border:1px solid #e5e7eb; 
          border-radius:12px; padding:24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
 h1 {{ margin: 0 0 8px 0; font-size: 28px; color: #111827; }}
 h2 {{ margin: 24px 0 12px 0; font-size: 20px; color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
 .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:16px; margin:16px 0; background: #f9fafb; }}
 .table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0; }}
 .table th {{ padding: 10px; border-bottom:2px solid #e5e7eb; text-align:left; background: #f3f4f6; font-weight: 600; }}
 .table td {{ padding: 8px; border-bottom:1px solid #f3f4f6; }}
 .muted {{ color:#6b7280; font-size:13px; margin: 4px 0 20px 0; }}
 .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 20px 0; }}
 .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
              padding: 20px; border-radius: 10px; text-align: center; }}
 .stat-value {{ font-size: 32px; font-weight: 700; margin: 8px 0; }}
 .stat-label {{ font-size: 13px; opacity: 0.9; }}
 pre {{ background: #f3f4f6; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
</style>
</head><body>
<div class="wrap">
<h1>ML Model Report</h1>
<div class="muted">Analysis ID: {analysis_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-label">Total Predictions</div>
    <div class="stat-value">{total_predictions}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Models Evaluated</div>
    <div class="stat-value">{num_models}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Confidence</div>
    <div class="stat-value">{f'{avg_confidence:.2f}' if avg_confidence is not None else 'N/A'}</div>
  </div>
</div>

<h2>Model Performance</h2>
<div class="card">{metrics_json}</div>

{''.join(f'<div class="card">{fig}</div>' for fig in figs)}

<h2>Recent Predictions</h2>
<div class="card">
  {table_html if table_html else '<p>No predictions available</p>'}
</div>

</div></body></html>"""
        
        output_path = os.path.join(out_dir, "ml_report.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        log.info(f"ML report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        log.error(f"Error generating ML report: {e}", exc_info=True)
        # Create minimal error report
        error_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>ML Report Error</title></head>
<body style="font-family: system-ui; padding: 20px;">
<h1>ML Report Generation Error</h1>
<p>An error occurred while generating the ML report:</p>
<pre style="background: #fee2e2; padding: 12px; border-radius: 8px;">{str(e)}</pre>
<p>Predictions received: {len(predictions)}</p>
<p>Metrics received: {len(metrics) if metrics else 0}</p>
</body></html>"""
        
        error_path = os.path.join(out_dir, "ml_report.html")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_html)
        
        return error_path


# ---------- MAIN ORCHESTRATION FUNCTION (FIXED) ----------
def build_all_artifacts(analysis_id: str, 
                       bottlenecks: Sequence[dict], 
                       predictions: Optional[Sequence[dict]] = None,
                       metrics: Optional[Dict[str,Any]] = None, 
                       out_dir: str = "out") -> Dict[str,str]:
    """
    One call from your analysis route to generate:
      - bottlenecks.html
      - ml_report.html (ENHANCED)
      - dashboard.html
    Returns dict of relative URLs to serve.
    """
    log.info(f"Building all artifacts for analysis {analysis_id}")
    log.info(f"Bottlenecks: {len(bottlenecks)}, Predictions: {len(predictions) if predictions else 0}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    ve = VisualizationEngine(out_dir)
    rg = ReportGenerator(out_dir)
    dm = DashboardManager(out_dir)

    # 1. Bottleneck Report
    try:
        r1 = rg.generate_bottleneck_report(
            analysis_id=analysis_id, 
            bottlenecks=bottlenecks, 
            metrics=metrics or {}
        )
        p1 = rg.export_report(r1, "bottlenecks.html")
        log.info(f"Bottleneck report generated: {p1}")
    except Exception as e:
        log.error(f"Bottleneck report generation failed: {e}", exc_info=True)
        p1 = ""

    # 2. ML Report (ENHANCED VERSION)
    p2 = ""
    if predictions:
        try:
            p2 = generate_ml_report_enhanced(
                analysis_id=analysis_id,
                predictions=predictions,
                metrics=metrics or {},
                out_dir=out_dir
            )
            log.info(f"ML report generated: {p2}")
        except Exception as e:
            log.error(f"ML report generation failed: {e}", exc_info=True)
            # Create minimal fallback report
            fallback_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>ML Report</title></head>
<body style="font-family: system-ui; padding: 20px;">
<h1>ML Model Report</h1>
<p>Predictions: {len(predictions)}</p>
<p>Report generation encountered an error: {str(e)}</p>
<h2>Raw Metrics</h2>
<pre>{json.dumps(metrics, indent=2) if metrics else 'No metrics'}</pre>
</body></html>"""
            
            p2 = os.path.join(out_dir, "ml_report.html")
            with open(p2, "w", encoding="utf-8") as f:
                f.write(fallback_html)
            log.info(f"Fallback ML report created: {p2}")
    else:
        log.warning("No predictions provided, skipping ML report")

    # 3. Dashboard
    p3 = ""
    try:
        comps = [
            ve.viz_bottleneck_severity(bottlenecks),
            ve.viz_bottleneck_distribution(bottlenecks),
            ve.viz_timeline(bottlenecks),
        ]
        dash = dm.createDashboard("AI Bottleneck Dashboard", comps)
        p3 = dm.publishDashboard(dash, "dashboard.html")
        log.info(f"Dashboard generated: {p3}")
    except Exception as e:
        log.error(f"Dashboard generation failed: {e}", exc_info=True)
        p3 = ""

    # Return URLs relative to /out/
    def make_url(path):
        if not path:
            return ""
        if not os.path.exists(path):
            log.warning(f"Expected artifact not found: {path}")
            return ""
        basename = os.path.basename(path)
        return f"/out/{basename}"
    
    result = {
        "bottlenecks": make_url(p1),
        "ml": make_url(p2),
        "dashboard": make_url(p3),
        "pdf": {}
    }
    
    log.info(f"Artifacts generated: {result}")
    return result


# ---------- PDF GENERATION (OPTIONAL) ----------
def _html_to_pdf(html_path: str, pdf_path: str) -> str:
    """
    Optional PDF generation using WeasyPrint.
    If WeasyPrint is not installed, this will silently fail and return empty string.
    """
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        return pdf_path
    except ImportError:
        log.warning("WeasyPrint not installed, skipping PDF generation")
        return ""
    except Exception as e:
        log.error(f"PDF generation failed: {e}")
        return ""


def build_all_artifacts_with_pdf(analysis_id: str,
                                 bottlenecks: Sequence[dict],
                                 predictions: Optional[Sequence[dict]] = None,
                                 metrics: Optional[Dict[str,Any]] = None,
                                 out_dir: str = "out") -> dict:
    """
    Extended version that also generates PDF versions of reports.
    Only use this if you have WeasyPrint installed.
    """
    # Generate HTML artifacts first
    paths = build_all_artifacts(
        analysis_id=analysis_id,
        bottlenecks=bottlenecks,
        predictions=predictions,
        metrics=metrics,
        out_dir=out_dir
    )
    
    # Convert to absolute file paths
    b_html = os.path.join(out_dir, os.path.basename(paths["bottlenecks"])) if paths.get("bottlenecks") else ""
    m_html = os.path.join(out_dir, os.path.basename(paths["ml"])) if paths.get("ml") else ""
    d_html = os.path.join(out_dir, os.path.basename(paths["dashboard"])) if paths.get("dashboard") else ""

    # Generate PDFs
    b_pdf = _html_to_pdf(b_html, os.path.join(out_dir, "bottlenecks.pdf")) if b_html else ""
    m_pdf = _html_to_pdf(m_html, os.path.join(out_dir, "ml_report.pdf")) if m_html else ""
    d_pdf = _html_to_pdf(d_html, os.path.join(out_dir, "dashboard.pdf")) if d_html else ""

    # Return same URL scheme, plus PDFs when available
    def as_url(p): 
        return f"/out/{os.path.basename(p)}" if p else ""

    return {
        **paths,
        "pdf": {
            "bottlenecks": as_url(b_pdf),
            "ml": as_url(m_pdf),
            "dashboard": as_url(d_pdf),
        }
    }