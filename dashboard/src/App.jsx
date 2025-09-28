import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

/** small styling helpers (no Tailwind required) **/
const box = {
  background: "#fff",
  borderRadius: 12,
  border: "1px solid #e5e7eb",
  padding: 16,
  boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
};
const h1 = { margin: 0, fontSize: 20 };
const row = { display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" };
const button = (primary) => ({
  padding: "10px 14px",
  borderRadius: 8,
  border: "1px solid " + (primary ? "#2563eb" : "#e5e7eb"),
  background: primary ? "#2563eb" : "#fff",
  color: primary ? "#fff" : "#111827",
  cursor: "pointer",
});
const select = { padding: 8, borderRadius: 8, border: "1px solid #e5e7eb" };
const input = { padding: 8, width: 160, borderRadius: 8, border: "1px solid #e5e7eb" };

export default function App() {
  // UI state
  const [activeTab, setActiveTab] = useState("dashboard"); // dashboard | analysis | reports | data
  const [domain, setDomain] = useState("manufacturing");   // manufacturing | logistics
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // Backend data
  const [bottlenecks, setBottlenecks] = useState([]); // from /analysis
  const [mlCard, setMlCard] = useState({});           // from /analysis
  const [artifacts, setArtifacts] = useState(null);   // from /analysis
  const [dbSummary, setDbSummary] = useState(null);   // from /dashboard
  const [whatIf, setWhatIf] = useState(null);         // from /whatif

  // What-If params
  const [trafficMultiplier, setTrafficMultiplier] = useState(1.0); // logistics
  const [latencyOffset, setLatencyOffset] = useState(0);           // manufacturing

  // Poll DB summary every 10s
  useEffect(() => {
    let stop = false;
    const poll = async () => {
      try {
        const r = await fetch(`${API_URL}/dashboard`);
        if (!stop && r.ok) setDbSummary(await r.json());
      } catch { /* ignore */ }
      if (!stop) setTimeout(poll, 10000);
    };
    poll();
    return () => { stop = true; };
  }, []);

  // Trigger full backend analysis
  const runAnalysis = async () => {
    setIsRunning(true);
    setErrorMsg("");
    setWhatIf(null);
    try {
      const r = await fetch(`${API_URL}/analysis/${domain}`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setBottlenecks(Array.isArray(data.bottlenecks) ? data.bottlenecks : []);
      setMlCard(data.ml_model_card || {});
      setArtifacts(data.artifacts || null);
      setActiveTab("dashboard");
    } catch (e) {
      setErrorMsg(`Run failed: ${String(e.message || e)}`);
    } finally {
      setIsRunning(false);
    }
  };

  // Trigger What-If
  const runWhatIf = async () => {
    setErrorMsg("");
    setWhatIf(null);
    try {
      const qs = domain === "logistics"
        ? `traffic_multiplier=${encodeURIComponent(trafficMultiplier)}`
        : `latency_offset_ms=${encodeURIComponent(latencyOffset)}`;
      const r = await fetch(`${API_URL}/whatif/${domain}?${qs}`);
      if (!r.ok) throw new Error(await r.text());
      setWhatIf(await r.json());
      setActiveTab("analysis");
    } catch (e) {
      setErrorMsg(`What-If failed: ${String(e.message || e)}`);
    }
  };

  /** Derived data for charts **/
  const topSeverity = useMemo(() => {
    const sorted = [...bottlenecks].sort((a, b) => (b.severity_score || 0) - (a.severity_score || 0));
    return sorted.slice(0, 12).map(r => ({
      entity: String(r.entity),
      severity: Number(r.severity_score || 0),
      type: r.bottleneck_type || "Unknown",
    }));
  }, [bottlenecks]);

  const typeDistribution = useMemo(() => {
    const counts = {};
    bottlenecks.forEach(b => { counts[b.bottleneck_type] = (counts[b.bottleneck_type] || 0) + 1; });
    const palette = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#f7b267", "#c4a7e7", "#84a59d"];
    return Object.entries(counts).map(([name, value], i) => ({ name, value, color: palette[i % palette.length] }));
  }, [bottlenecks]);

  const clsMetrics = useMemo(() => {
    const entry = (mlCard.manufacturing || mlCard.logistics || {});
    const m = entry.metrics || {};
    return [
      { metric: "Accuracy", value: Number(m.accuracy || 0) },
      { metric: "Precision", value: Number(m.precision || 0) },
      { metric: "Recall", value: Number(m.recall || 0) },
      { metric: "F1", value: Number(m.f1 || 0) },
    ];
  }, [mlCard]);

  const timeline = useMemo(() => {
    return [...bottlenecks].slice(0, 20).reverse().map(b => ({
      t: String(b.detected_at || "").replace("T", " ").slice(0, 19),
      s: Number(b.severity_score || 0),
    }));
  }, [bottlenecks]);

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(120deg,#eff6ff,#eef2ff)" }}>
      {/* Header */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 style={h1}>AI-Powered Bottleneck Detection</h1>
            <div style={{ color: "#6b7280", fontSize: 13 }}>Manufacturing & Logistics — Backend: {API_URL}</div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => setActiveTab("dashboard")} style={button(activeTab==="dashboard")}>Dashboard</button>
            <button onClick={() => setActiveTab("analysis")} style={button(activeTab==="analysis")}>Analysis</button>
            <button onClick={() => setActiveTab("reports")} style={button(activeTab==="reports")}>Reports</button>
            <button onClick={() => setActiveTab("data")} style={button(activeTab==="data")}>Data</button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ maxWidth: 1200, margin: "16px auto", padding: "0 16px" }}>
        <div style={{ ...box, ...row, justifyContent: "space-between" }}>
          <div style={row}>
            <div>
              <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Data Type</div>
              <select value={domain} onChange={(e) => setDomain(e.target.value)} style={select}>
                <option value="manufacturing">Manufacturing</option>
                <option value="logistics">Logistics</option>
              </select>
            </div>

            {domain === "logistics" ? (
              <div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Traffic Multiplier (What-If)</div>
                <input
                  type="number" min={0.5} max={2} step={0.1}
                  value={trafficMultiplier} onChange={(e) => setTrafficMultiplier(Number(e.target.value))}
                  style={input}
                />
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Latency Offset (ms) (What-If)</div>
                <input
                  type="number" min={-50} max={200} step={1}
                  value={latencyOffset} onChange={(e) => setLatencyOffset(Number(e.target.value))}
                  style={input}
                />
              </div>
            )}
          </div>

          <div style={row}>
            <button onClick={runAnalysis} disabled={isRunning} style={button(true)}>
              {isRunning ? "Running…" : "Run Analysis"}
            </button>
            <button onClick={runWhatIf} style={button(false)}>Run What-If</button>
          </div>
        </div>

        {errorMsg && (
          <div style={{ ...box, borderLeft: "4px solid #dc2626", color: "#991b1b", background: "#fee2e2" }}>
            {errorMsg}
          </div>
        )}

        {/* DASHBOARD TAB */}
        {activeTab === "dashboard" && (
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr" }}>
            {/* Metrics */}
            <div style={{ ...box, gridColumn: "1 / -1", display: "grid", gap: 12, gridTemplateColumns: "repeat(4, 1fr)" }}>
              <Metric title="Active Bottlenecks" value={bottlenecks.length} />
              <Metric
                title="System Efficiency"
                value={topSeverity.length ? `${(100 - (topSeverity[0]?.severity || 0)).toFixed(1)}%` : "--"}
                hint="Lower severity → better"
              />
              <Metric
                title="Model F1"
                value={
                  (clsMetrics.find(m => m.metric === "F1")?.value ?? 0).toFixed(2)
                }
              />
              <Metric title="Artifacts Ready" value={artifacts ? "Yes" : "No"} />
            </div>

            {/* Top severity */}
            <div style={box}>
              <h3 style={{ margin: "0 0 12px 0" }}>Bottleneck Severity (Top 12)</h3>
              <div style={{ width: "100%", height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={topSeverity}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="entity" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="severity" fill="#2563eb" radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Distribution */}
            <div style={box}>
              <h3 style={{ margin: "0 0 12px 0" }}>Bottleneck Types Distribution</h3>
              <div style={{ width: "100%", height: 300 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={typeDistribution} dataKey="value" cx="50%" cy="50%" outerRadius={100}
                         label={({ name, percent }) => `${name}: ${(percent*100).toFixed(0)}%`}>
                      {typeDistribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Timeline */}
            <div style={{ ...box, gridColumn: "1 / -1" }}>
              <h3 style={{ margin: "0 0 12px 0" }}>Recent Detections Timeline</h3>
              <div style={{ width: "100%", height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="s" stroke="#7c3aed" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ANALYSIS TAB */}
        {activeTab === "analysis" && (
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr" }}>
            <div style={box}>
              <h3 style={{ marginTop: 0 }}>Current Bottlenecks (latest run)</h3>
              {!bottlenecks.length ? (
                <div style={{ color: "#6b7280", fontSize: 14 }}>Run analysis to see detections.</div>
              ) : (
                <div style={{ display: "grid", gap: 8 }}>
                  {bottlenecks.map((b, idx) => (
                    <div key={idx} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <strong>{String(b.entity)}</strong>
                        <span style={{ fontSize: 12, color: "#374151" }}>{b.bottleneck_type}</span>
                      </div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>Reason: {b.reason || "—"}</div>
                      <div style={{ marginTop: 8 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                          <span>Severity</span><span>{Number(b.severity_score || 0).toFixed(1)}%</span>
                        </div>
                        <div style={{ background: "#e5e7eb", height: 8, borderRadius: 999, marginTop: 4 }}>
                          <div style={{
                            width: `${Number(b.severity_score || 0)}%`,
                            height: 8, borderRadius: 999, background: "#ef4444"
                          }} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={box}>
              <h3 style={{ marginTop: 0 }}>What-If Delta</h3>
              {!whatIf?.delta?.length ? (
                <div style={{ color: "#6b7280", fontSize: 14 }}>Run a What-If to see impact.</div>
              ) : (
                <div style={{ maxHeight: 360, overflow: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                    <thead>
                      <tr style={{ background: "#f9fafb" }}>
                        <th style={th}>Entity</th>
                        <th style={th}>Type</th>
                        <th style={{ ...th, textAlign: "right" }}>Before</th>
                        <th style={{ ...th, textAlign: "right" }}>After</th>
                        <th style={{ ...th, textAlign: "right" }}>Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {whatIf.delta.map((d, i) => (
                        <tr key={i}>
                          <td style={td}>{d.entity}</td>
                          <td style={td}>{d.bottleneck_type}</td>
                          <td style={{ ...td, textAlign: "right" }}>{d.severity_before}</td>
                          <td style={{ ...td, textAlign: "right" }}>{d.severity_after}</td>
                          <td style={{
                            ...td, textAlign: "right",
                            color: d.delta >= 0 ? "#b91c1c" : "#065f46"
                          }}>
                            {d.delta}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Model metrics */}
            <div style={{ ...box, gridColumn: "1 / -1" }}>
              <h3 style={{ marginTop: 0 }}>Model Metrics (latest run)</h3>
              <div style={{ width: "100%", height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={clsMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#2563eb" radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* REPORTS TAB */}
        {activeTab === "reports" && (
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <ReportCard
              title="Bottleneck Report"
              href={artifacts?.static_urls?.bottlenecks ? `${API_URL}${artifacts.static_urls.bottlenecks}` : "#"}
            />
            <ReportCard
              title="ML Report"
              href={artifacts?.static_urls?.ml ? `${API_URL}${artifacts.static_urls.ml}` : "#"}
            />
            <ReportCard
              title="Interactive Dashboard (HTML)"
              href={`${API_URL}/out/dashboard.html`}
            />
            <div style={{ ...box, gridColumn: "1 / -1" }}>
              <h4 style={{ marginTop: 0 }}>Inline Dashboard Preview</h4>
              <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, overflow: "hidden" }}>
                <iframe
                  title="backend-dashboard"
                  src={`${API_URL}/out/dashboard.html`}
                  style={{ width: "100%", height: 600, border: "none" }}
                />
              </div>
            </div>
          </div>
        )}

        {/* DATA TAB */}
        {activeTab === "data" && (
          <div style={{ ...box }}>
            <h3 style={{ marginTop: 0 }}>Data & Pipeline</h3>
            <p style={{ color: "#6b7280" }}>
              CSVs are read by the backend. Use the “Run Analysis” button to re-ingest and compute artifacts.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <a href={`${API_URL}/analysis/${domain}`} target="_blank" rel="noreferrer" style={{ ...button(true), textDecoration: "none" }}>
                Re-run {domain} analysis
              </a>
              <a href={`${API_URL}/out/dashboard.html`} target="_blank" rel="noreferrer" style={{ ...button(false), textDecoration: "none" }}>
                View backend dashboard
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/** tiny components **/

function Metric({ title, value, hint }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, padding: 12 }}>
      <div style={{ color: "#6b7280", fontSize: 12 }}>{title}</div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{String(value ?? "--")}</div>
      {hint && <div style={{ color: "#9ca3af", fontSize: 12 }}>{hint}</div>}
    </div>
  );
}

function ReportCard({ title, href }) {
  return (
    <div style={box}>
      <h4 style={{ marginTop: 0 }}>{title}</h4>
      <p style={{ color: "#6b7280", fontSize: 14 }}>Open the latest backend-generated HTML.</p>
      <div style={{ display: "flex", gap: 8 }}>
        <a href={href} target="_blank" rel="noreferrer" style={{ ...button(true), textDecoration: "none" }}>View</a>
        <a href={href} target="_blank" rel="noreferrer" download style={{ ...button(false), textDecoration: "none" }}>Download</a>
      </div>
    </div>
  );
}

const th = { padding: 8, borderBottom: "1px solid #e5e7eb", textAlign: "left" };
const td = { padding: 8, borderBottom: "1px solid #f3f4f6" };
