import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from "recharts";

// Dynamic API URL detection
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? "http://127.0.0.1:8000"
  : `${window.location.protocol}//${window.location.hostname}:8000`;

const box = { background:"#fff", borderRadius:12, border:"1px solid #e5e7eb", padding:16, boxShadow:"0 1px 2px rgba(0,0,0,0.05)" };
const button = (primary, disabled = false) => ({ 
  padding:"10px 14px", 
  borderRadius:8, 
  border:"1px solid "+(primary?"#2563eb":"#e5e7eb"), 
  background: disabled ? "#d1d5db" : (primary?"#2563eb":"#fff"), 
  color: disabled ? "#6b7280" : (primary?"#fff":"#111827"), 
  cursor: disabled ? "not-allowed" : "pointer", 
  fontWeight:500,
  opacity: disabled ? 0.6 : 1,
  transition: "all 0.2s"
});
const select = { padding:8, borderRadius:8, border:"1px solid #e5e7eb", fontSize:14 };
const input = { padding:8, width:160, borderRadius:8, border:"1px solid #e5e7eb", fontSize:14 };
const th = { padding:8, borderBottom:"2px solid #e5e7eb", textAlign:"left", fontWeight:600 };
const td = { padding:8, borderBottom:"1px solid #f3f4f6" };

async function fetchJson(url, { timeoutMs = 10000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: ctrl.signal });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Request timeout');
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function Badge({ text, tone="neutral" }) {
  const colors = {
    neutral:{ bg:"#f3f4f6", color:"#111827" },
    good:{ bg:"#dcfce7", color:"#14532d" },
    warn:{ bg:"#fef9c3", color:"#854d0e" },
    bad:{ bg:"#fee2e2", color:"#991b1b" },
    info:{ bg:"#e0ecff", color:"#1e40af" },
  }[tone];
  return <span style={{ fontSize:12, padding:"4px 10px", borderRadius:999, background:colors.bg, color:colors.color, fontWeight:500 }}>{text}</span>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [domain, setDomain] = useState("logistics");
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [loadingDash, setLoadingDash] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState("checking");

  const [bottlenecks, setBottlenecks] = useState([]);
  const [mlCard, setMlCard] = useState({});
  const [artifacts, setArtifacts] = useState(null);
  const [dbSummary, setDbSummary] = useState(null);
  const [whatIf, setWhatIf] = useState(null);

  const [trafficMultiplier, setTrafficMultiplier] = useState(1.0);
  const [latencyOffset, setLatencyOffset] = useState(0);
  const [pollInterval, setPollInterval] = useState(10000);

  // Connection health check
  const checkConnection = useCallback(async () => {
    try {
      await fetchJson(`${API_URL}/dashboard`, { timeoutMs: 3000 });
      setConnectionStatus("connected");
      return true;
    } catch (error) {
      setConnectionStatus("disconnected");
      return false;
    }
  }, []);

  // Intelligent polling based on pipeline activity
  useEffect(() => {
    if (dbSummary?.pipeline) {
      const hasActive = Object.values(dbSummary.pipeline).some(s => s === "active");
      setPollInterval(hasActive ? 3000 : 10000); // Fast poll when active
    }
  }, [dbSummary?.pipeline]);

  // Dashboard polling with connection check
  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const data = await fetchJson(`${API_URL}/dashboard`, { timeoutMs: 5000 });
        setDbSummary(data);
        setLoadingDash(false);
        setConnectionStatus("connected");
      } catch (error) {
        console.error("Dashboard fetch error:", error);
        setConnectionStatus("error");
        setLoadingDash(false);
      }
    };

    checkConnection().then(connected => {
      if (connected) loadDashboard();
    });
    
    const interval = setInterval(loadDashboard, pollInterval);
    return () => clearInterval(interval);
  }, [pollInterval, checkConnection]);

  const runAnalysis = async () => {
    setIsRunning(true);
    setErrorMsg("");
    setSuccessMsg("");
    setWhatIf(null);

    const timeoutMs = 120000;
    try {
      const url = `${API_URL}/analysis/${domain}?auto_train=1`;
      const data = await fetchJson(url, { timeoutMs });
      
      setBottlenecks(Array.isArray(data.bottlenecks) ? data.bottlenecks : []);
      setMlCard(data.ml_model_card || {});
      setArtifacts((data.artifacts && Object.keys(data.artifacts).length>0) ? data.artifacts : null);
      setSuccessMsg(`Analysis completed — found ${data.bottlenecks?.length || 0} bottlenecks.`);
      setActiveTab("dashboard");
      
      // Refresh dashboard
      const dashData = await fetchJson(`${API_URL}/dashboard`, { timeoutMs: 5000 });
      setDbSummary(dashData);
    } catch (error) {
      console.error("runAnalysis error:", error);
      setErrorMsg(`Analysis failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const runWhatIf = async () => {
    setErrorMsg("");
    setSuccessMsg("");
    setWhatIf(null);

    try {
      const params = new URLSearchParams();
      params.append("traffic_multiplier", String(trafficMultiplier));
      params.append("latency_offset_ms", String(latencyOffset));

      const data = await fetchJson(`${API_URL}/whatif/${domain}?${params.toString()}`, { timeoutMs: 30000 });
      setWhatIf(data);
      setSuccessMsg(`What-If analysis completed.`);
      setActiveTab("analysis");
    } catch (error) {
      console.error("whatif error:", error);
      setErrorMsg(`What-If failed: ${error.message}`);
    }
  };

  const topSeverity = useMemo(() => {
    const sorted = [...bottlenecks].sort((a, b) => (b.severity_score || 0) - (a.severity_score || 0));
    return sorted.slice(0, 12).map(r => ({
      entity: String(r.entity || "Unknown").slice(0, 20),
      severity: Number(r.severity_score || 0),
      type: String(r.bottleneck_type || "Unknown").slice(0, 30),
    }));
  }, [bottlenecks]);

  const typeDistribution = useMemo(() => {
    const counts = {};
    bottlenecks.forEach(b => {
      const type = b.bottleneck_type || "Unknown";
      counts[type] = (counts[type] || 0) + 1;
    });
    const palette = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6"];
    return Object.entries(counts).map(([name, value], i) => ({ 
      name: name.slice(0, 30), 
      value, 
      color: palette[i % palette.length] 
    }));
  }, [bottlenecks]);

  const mlMetrics = useMemo(() => {
    const cards = mlCard?.cards || mlCard;
    const metrics = cards?.eta?.metrics?.test || cards?.delay?.metrics?.test || 
                   cards?.speed?.metrics?.test || cards?.error?.metrics?.test || {};
    
    return [
      { metric: "MAE", value: Number(metrics.MAE || 0) },
      { metric: "R2", value: Number(metrics.R2 || 0) },
      { metric: "AUC", value: Number(metrics.AUC || 0) },
      { metric: "F1", value: Number(metrics.F1 || 0) },
    ].filter(m => Number(m.value) > 0);
  }, [mlCard]);

  const timeline = useMemo(() => {
    return [...bottlenecks]
      .slice(0, 50)
      .reverse()
      .map((b, i) => ({
        index: i + 1,
        t: String(b.detected_at || "").split("T")[1]?.slice(0, 8) || `T${i}`,
        severity: Number(b.severity_score || 0),
      }));
  }, [bottlenecks]);

  // Real data from API
  const sources = dbSummary?.data_sources || [];
  const pipeline = dbSummary?.pipeline || {};
  const predsCount = dbSummary?.predictions_count || 0;
  const totalRuns = dbSummary?.total_runs || 0;
  const totalBottlenecks = dbSummary?.total_bottlenecks || 0;
  const latestRun = dbSummary?.latest_run || {};

  const artifactUrls = useMemo(() => {
    if (!artifacts) return {};
    const out = {};
    ['bottlenecks', 'ml', 'dashboard'].forEach(key => {
      if (artifacts[key]) {
        const url = artifacts[key];
        out[key] = url.startsWith("http") ? url : `${API_URL}${url.startsWith("/") ? "" : "/"}${url}`;
      }
    });
    return out;
  }, [artifacts]);

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" }}>
      <div style={{ background: "rgba(255,255,255,0.95)", backdropFilter: "blur(10px)", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "16px 24px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <h1 style={{ margin:0, fontSize:28, fontWeight:700, background:"linear-gradient(135deg, #667eea, #764ba2)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
                AI Bottleneck Detection System
              </h1>
              <div style={{ margin:"4px 0 0 0", color:"#6b7280", fontSize:14, display:"flex", gap:12, alignItems:"center" }}>
                <span>Real-time supply chain intelligence</span>
                <Badge 
                  text={connectionStatus === "connected" ? "Connected" : connectionStatus === "checking" ? "Connecting..." : "Disconnected"} 
                  tone={connectionStatus === "connected" ? "good" : connectionStatus === "checking" ? "info" : "bad"} 
                />
                {latestRun?.status && (
                  <Badge text={`Last Run: ${latestRun.status}`} tone={latestRun.status === "success" ? "good" : "warn"} />
                )}
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => setActiveTab("dashboard")} style={button(activeTab==="dashboard")}>Dashboard</button>
              <button onClick={() => setActiveTab("analysis")} style={button(activeTab==="analysis")}>Analysis</button>
              <button onClick={() => setActiveTab("reports")} style={button(activeTab==="reports")}>Reports</button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "24px" }}>
        <div style={{ ...box, marginBottom: 16, background: "rgba(255,255,255,0.95)" }}>
          <div style={{ display:"flex", gap:16, justifyContent: "space-between", flexWrap:"wrap", alignItems:"center" }}>
            <div style={{ display:"flex", gap:16, flexWrap:"wrap", alignItems:"center" }}>
              <div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6, fontWeight: 500 }}>Data Domain</div>
                <select 
                  value={domain} 
                  onChange={(e) => setDomain(e.target.value)} 
                  style={select}
                  disabled={isRunning}
                >
                  <option value="manufacturing">Manufacturing</option>
                  <option value="logistics">Logistics</option>
                </select>
              </div>

              {domain === "logistics" ? (
                <div>
                  <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6, fontWeight: 500 }}>
                    Traffic Multiplier ({trafficMultiplier.toFixed(1)}x)
                  </div>
                  <input type="number" min={0.5} max={2} step={0.1}
                    value={trafficMultiplier} 
                    onChange={(e) => setTrafficMultiplier(Number(e.target.value))}
                    style={input}
                    disabled={isRunning}
                  />
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6, fontWeight: 500 }}>
                    Latency Offset ({latencyOffset}ms)
                  </div>
                  <input type="number" min={-50} max={200} step={5}
                    value={latencyOffset} 
                    onChange={(e) => setLatencyOffset(Number(e.target.value))}
                    style={input}
                    disabled={isRunning}
                  />
                </div>
              )}
            </div>

            <div style={{ display:"flex", gap:8 }}>
              <button 
                onClick={runAnalysis} 
                disabled={isRunning || connectionStatus !== "connected"} 
                style={button(true, isRunning || connectionStatus !== "connected")}
              >
                {isRunning ? "Running... (this can take a while)" : "Run Analysis"}
              </button>
              <button 
                onClick={runWhatIf} 
                disabled={isRunning || connectionStatus !== "connected"}
                style={button(false, isRunning || connectionStatus !== "connected")}
              >
                Run What-If
              </button>
            </div>
          </div>
        </div>

        {errorMsg && (
          <div style={{ ...box, marginBottom: 16, borderLeft: "4px solid #dc2626", color: "#991b1b", background: "#fee2e2" }}>
            <strong>Error:</strong> {errorMsg}
          </div>
        )}
        {successMsg && (
          <div style={{ ...box, marginBottom: 16, borderLeft: "4px solid #059669", color: "#065f46", background: "#d1fae5" }}>
            ✓ {successMsg}
          </div>
        )}

        {activeTab === "dashboard" && (
          <div style={{ display: "grid", gap: 16 }}>
            <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
              <Metric title="Active Bottlenecks" value={bottlenecks.length || totalBottlenecks} color="#ef4444" loading={loadingDash} />
              <Metric title="Total Analysis Runs" value={totalRuns} color="#3b82f6" loading={loadingDash} />
              <Metric title="ML Predictions" value={predsCount} color="#8b5cf6" loading={loadingDash} />
              <Metric title="Data Sources" value={sources.filter(s => s.status === "connected").length} color="#10b981" loading={loadingDash} />
            </div>

            <div style={{ display: "grid", gap: 16, gridTemplateColumns: "2fr 1fr" }}>
              <div style={box}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Top Bottlenecks by Severity</h3>
                {topSeverity.length === 0 ? (
                  <div style={{ color:"#6b7280", padding:40, textAlign:"center" }}>
                    No bottlenecks detected. Run an analysis to see results.
                  </div>
                ) : (
                  <div style={{ width: "100%", height: 320 }}>
                    <ResponsiveContainer>
                      <BarChart data={topSeverity}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="entity" tick={{ fontSize: 11 }} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="severity" fill="#ef4444" radius={[8,8,0,0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div style={box}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Distribution by Type</h3>
                {typeDistribution.length === 0 ? (
                  <div style={{ color:"#6b7280", padding:40, textAlign:"center", fontSize:14 }}>
                    No data
                  </div>
                ) : (
                  <div style={{ width: "100%", height: 320 }}>
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie 
                          data={typeDistribution} 
                          dataKey="value" 
                          cx="50%" 
                          cy="50%" 
                          outerRadius={90}
                          label={({ percent }) => `${(percent*100).toFixed(0)}%`}
                        >
                          {typeDistribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                        </Pie>
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>

            <div style={box}>
              <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Detection Timeline</h3>
              {timeline.length === 0 ? (
                <div style={{ color:"#6b7280", padding:40, textAlign:"center" }}>
                  No timeline data available
                </div>
              ) : (
                <div style={{ width: "100%", height: 280 }}>
                  <ResponsiveContainer>
                    <LineChart data={timeline}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="t" tick={{ fontSize: 11 }} />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="severity" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr" }}>
              <div style={box}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Data Sources</h3>
                {loadingDash ? (
                  <div style={{ color:"#6b7280", textAlign:"center", padding:20 }}>Loading...</div>
                ) : sources.length === 0 ? (
                  <div style={{ color:"#6b7280", fontSize:14 }}>No sources configured</div>
                ) : (
                  <div style={{ display:"grid", gap:10 }}>
                    {sources.map((s, i) => (
                      <div key={i} style={{ border:"1px solid #e5e7eb", borderRadius:8, padding:12, background:"#f9fafb" }}>
                        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6 }}>
                          <strong style={{ fontSize:14 }}>{s.name}</strong>
                          <Badge text={s.status || "unknown"} tone={s.status==="connected"?"good":"warn"} />
                        </div>
                        <div style={{ fontSize:12, color:"#6b7280" }}>
                          Records: {s.records?.toLocaleString() || "—"}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={box}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Pipeline Status</h3>
                {loadingDash ? (
                  <div style={{ color:"#6b7280", textAlign:"center", padding:20 }}>Loading...</div>
                ) : (
                  <div style={{ display:"grid", gap:10 }}>
                    {[
                      ["Ingestion", pipeline.ingestion],
                      ["Cleaning", pipeline.cleaning],
                      ["Feature Engineering", pipeline.feature_engineering],
                      ["Training", pipeline.training],
                    ].map(([label, status]) => (
                      <div key={label} style={{ display:"flex", justifyContent:"space-between", border:"1px solid #e5e7eb", borderRadius:8, padding:12, background:"#f9fafb" }}>
                        <span style={{ fontSize:14 }}>{label}</span>
                        <Badge
                          text={status || "idle"}
                          tone={
                            status==="active" ? "good" : 
                            status==="scheduled" ? "info" : 
                            status==="idle" ? "neutral" : "warn"
                          }
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "analysis" && (
          <div style={{ display: "grid", gap: 16 }}>
            <div style={box}>
              <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Current Bottlenecks</h3>
              {bottlenecks.length === 0 ? (
                <div style={{ color:"#6b7280", padding:40, textAlign:"center" }}>
                  No bottlenecks detected. Click Run Analysis to start.
                </div>
              ) : (
                <div style={{ display:"grid", gap:12, gridTemplateColumns:"repeat(auto-fill, minmax(320px, 1fr))" }}>
                  {bottlenecks.map((b, idx) => (
                    <div key={idx} style={{ 
                      border:"2px solid #e5e7eb", 
                      borderRadius:12, 
                      padding:16,
                      background: `linear-gradient(135deg, ${
                        b.severity_score >= 80 ? "#fee2e2" : 
                        b.severity_score >= 50 ? "#fef3c7" : "#dbeafe"
                      }, #fff)`
                    }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                        <strong style={{ fontSize:16 }}>{String(b.entity).slice(0, 25)}</strong>
                        <Badge text={String(b.bottleneck_type).slice(0, 20)} tone="info" />
                      </div>
                      <div style={{ fontSize:13, color:"#374151", marginBottom:12 }}>
                        {String(b.reason || b.description || "—").slice(0, 80)}
                      </div>
                      <div>
                        <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, marginBottom:4 }}>
                          <span style={{ fontWeight:500 }}>Severity</span>
                          <span style={{ fontWeight:600 }}>{Number(b.severity_score || 0).toFixed(1)}%</span>
                        </div>
                        <div style={{ background:"#e5e7eb", height:10, borderRadius:999 }}>
                          <div style={{ 
                            width:`${Math.min(Number(b.severity_score || 0), 100)}%`, 
                            height:10, 
                            borderRadius:999, 
                            background: b.severity_score >= 80 ? "#ef4444" : 
                                       b.severity_score >= 50 ? "#f59e0b" : "#10b981",
                            transition: "width 0.3s ease"
                          }} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {whatIf && (
              <div style={box}>
                <h3 style={{ margin:"0 0 16px 0", fontSize:18 }}>What-If Scenario Results</h3>
                <div style={{ marginBottom:16, padding:12, background:"#f0f9ff", borderRadius:8, border:"1px solid #bae6fd" }}>
                  <div style={{ fontSize:13, color:"#0c4a6e" }}>
                    <strong>Parameters:</strong> {
                      domain === "logistics" 
                        ? `Traffic × ${whatIf.params?.traffic_multiplier || 1.0}`
                        : `Latency + ${whatIf.params?.latency_offset_ms || 0}ms`
                    }
                  </div>
                </div>
                {!whatIf.delta?.length ? (
                  <div style={{ color:"#6b7280", padding:20, textAlign:"center" }}>No changes detected</div>
                ) : (
                  <div style={{ overflowX:"auto" }}>
                    <table style={{ width:"100%", borderCollapse:"collapse", fontSize:14 }}>
                      <thead>
                        <tr style={{ background:"#f9fafb" }}>
                          <th style={th}>Entity</th>
                          <th style={th}>Type</th>
                          <th style={{ ...th, textAlign:"right" }}>Before</th>
                          <th style={{ ...th, textAlign:"right" }}>After</th>
                          <th style={{ ...th, textAlign:"right" }}>Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {whatIf.delta.map((d, i) => (
                          <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                            <td style={td}>{String(d.entity).slice(0, 30)}</td>
                            <td style={td}>{String(d.bottleneck_type).slice(0, 30)}</td>
                            <td style={{ ...td, textAlign:"right", fontWeight:600 }}>{d.severity_before}</td>
                            <td style={{ ...td, textAlign:"right", fontWeight:600 }}>{d.severity_after}</td>
                            <td style={{ 
                              ...td, 
                              textAlign:"right", 
                              fontWeight:700,
                              color: d.delta > 0 ? "#dc2626" : d.delta < 0 ? "#059669" : "#6b7280"
                            }}>
                              {d.delta > 0 ? "+" : ""}{d.delta}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {mlMetrics.length > 0 && (
              <div style={box}>
                <h3 style={{ margin:"0 0 16px 0", fontSize:18 }}>Model Performance Metrics</h3>
                <div style={{ width:"100%", height:280 }}>
                  <ResponsiveContainer>
                    <BarChart data={mlMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="metric" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#3b82f6" radius={[8,8,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "reports" && (
          <div style={{ display:"grid", gap:16, gridTemplateColumns:"repeat(auto-fill, minmax(300px, 1fr))" }}>
            <ReportCard 
              title="Bottleneck Report" 
              artifactUrl={artifactUrls.bottlenecks}
              fallbackFile="bottlenecks.html"
              desc="Detailed bottleneck analysis report"
            />
            <ReportCard 
              title="ML Report" 
              artifactUrl={artifactUrls.ml}
              fallbackFile={domain === "manufacturing" ? "manufacturing_ml_report.html" : "logistics_ml_report.html"}
              desc="Model performance and predictions"
            />
            <ReportCard 
              title="Dashboard (HTML)" 
              artifactUrl={artifactUrls.dashboard}
              fallbackFile="dashboard.html"
              desc="Interactive backend dashboard"
            />
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ title, value, color = "#3b82f6", loading = false }) {
  return (
    <div style={{ 
      border:"2px solid #e5e7eb", 
      borderRadius:12, 
      padding:20,
      background:"#fff",
      boxShadow:"0 2px 4px rgba(0,0,0,0.05)",
      position:"relative"
    }}>
      <div style={{ color:"#6b7280", fontSize:13, marginBottom:8, fontWeight:500 }}>{title}</div>
      {loading ? (
        <div style={{ fontSize:32, fontWeight:700, color:"#d1d5db" }}>...</div>
      ) : (
        <div style={{ fontSize:32, fontWeight:700, color }}>{String(value)}</div>
      )}
    </div>
  );
}

function ReportCard({ title, artifactUrl, fallbackFile, desc }) {
  const [checkStatus, setCheckStatus] = useState("checking");
  const [finalUrl, setFinalUrl] = useState(null);
  
  useEffect(() => {
    const checkAvailability = async () => {
      let urlToCheck = artifactUrl;
      if (!urlToCheck && fallbackFile) {
        urlToCheck = `${API_URL}/out/${fallbackFile}`;
      }
      
      if (!urlToCheck) {
        setCheckStatus("missing");
        return;
      }

      try {
        const response = await fetch(urlToCheck, { method: 'HEAD' });
        if (response.ok) {
          setFinalUrl(urlToCheck);
          setCheckStatus("available");
        } else {
          setCheckStatus("missing");
        }
      } catch (error) {
        console.error(`Failed to check ${urlToCheck}:`, error);
        setCheckStatus("missing");
      }
    };

    checkAvailability();
  }, [artifactUrl, fallbackFile]);

  const isAvailable = checkStatus === "available" && finalUrl;
  
  const handleView = () => {
    if (finalUrl) {
      window.open(finalUrl, '_blank');
    }
  };

  const handleDownload = async () => {
    if (!finalUrl) return;
    try {
      const response = await fetch(finalUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fallbackFile || 'report.html';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };
  
  return (
    <div style={{ 
      ...box, 
      border:"2px solid #e5e7eb",
      transition:"all 0.2s",
      opacity: isAvailable ? 1 : 0.6
    }}>
      <h4 style={{ margin:"0 0 8px 0", fontSize:16 }}>{title}</h4>
      <p style={{ color:"#6b7280", fontSize:13, marginBottom:16 }}>{desc}</p>
      
      {checkStatus === "checking" && (
        <div style={{ 
          color:"#6b7280", 
          fontSize:13, 
          marginBottom:16,
          padding:8,
          background:"#f3f4f6",
          borderRadius:6
        }}>
          Checking availability...
        </div>
      )}
      
      {checkStatus === "missing" && (
        <div style={{ 
          color:"#dc2626", 
          fontSize:13, 
          marginBottom:16,
          padding:8,
          background:"#fee2e2",
          borderRadius:6
        }}>
          Report not generated yet. Run analysis to generate.
        </div>
      )}
      
      {isAvailable && (
        <div style={{ 
          color:"#059669", 
          fontSize:13, 
          marginBottom:16,
          padding:8,
          background:"#d1fae5",
          borderRadius:6
        }}>
          Report ready
        </div>
      )}
      
      <div style={{ display:"flex", gap:8 }}>
        <button 
          onClick={handleView}
          disabled={!isAvailable}
          style={{ 
            ...button(isAvailable, !isAvailable), 
            flex:1
          }}
        >
          View
        </button>
        <button 
          onClick={handleDownload}
          disabled={!isAvailable}
          style={{ 
            ...button(false, !isAvailable), 
            flex:1
          }}
        >
          Download
        </button>
      </div>
    </div>
  );
}