import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";
import TopBarPro from "./components/TopBarPro";
import ChartPro from "./components/ChartPro";
import ModelsPro from "./components/ModelsPro";
import FinalSignalPro from "./components/FinalSignalPro";
import BacktestPanel from "./components/BacktestPanel";
import ShapPanel from "./components/ShapPanel";
import RiskPanel from "./components/RiskPanel";

const API = "http://127.0.0.1:8000";

const TIMEZONES = [
  { label: "Athens",    tz: "Europe/Athens",     flag: "🇬🇷", open: "10:00", close: "18:20" },
  { label: "London",    tz: "Europe/London",      flag: "🇬🇧", open: "08:00", close: "16:30" },
  { label: "New York",  tz: "America/New_York",   flag: "🇺🇸", open: "09:30", close: "16:00" },
  { label: "Chicago",   tz: "America/Chicago",    flag: "🇺🇸", open: "08:30", close: "15:00" },
  { label: "Frankfurt", tz: "Europe/Berlin",      flag: "🇩🇪", open: "09:00", close: "17:30" },
  { label: "Paris",     tz: "Europe/Paris",       flag: "🇫🇷", open: "09:00", close: "17:30" },
  { label: "Tokyo",     tz: "Asia/Tokyo",         flag: "🇯🇵", open: "09:00", close: "15:30" },
  { label: "Hong Kong", tz: "Asia/Hong_Kong",     flag: "🇭🇰", open: "09:30", close: "16:00" },
  { label: "Sydney",    tz: "Australia/Sydney",   flag: "🇦🇺", open: "10:00", close: "16:00" },
  { label: "Dubai",     tz: "Asia/Dubai",         flag: "🇦🇪", open: "10:00", close: "14:00" },
  { label: "UTC",       tz: "UTC",                flag: "🌐", open: null,    close: null    },
];

function isMarketOpen(tzObj, now) {
  if (!tzObj.open) return false;
  const local = new Date(now.toLocaleString("en-US", { timeZone: tzObj.tz }));
  const day = local.getDay();
  if (day === 0 || day === 6) return false;
  const total = local.getHours() * 60 + local.getMinutes();
  const [oh, om] = tzObj.open.split(":").map(Number);
  const [ch, cm] = tzObj.close.split(":").map(Number);
  return total >= oh * 60 + om && total < ch * 60 + cm;
}

function SignalIcon({ label, color }) {
  const c = color || "#eab308";
  const map = {
    "Benzinga": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><rect x="3" y="4" width="18" height="16" rx="2" stroke={c} strokeWidth="1.8"/><path d="M7 8h10M7 12h7M7 16h5" stroke={c} strokeWidth="1.8" strokeLinecap="round"/></svg>),
    "CNBC": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><circle cx="12" cy="12" r="9" stroke={c} strokeWidth="1.8"/><path d="M12 6v6l4 2" stroke={c} strokeWidth="1.8" strokeLinecap="round"/></svg>),
    "MarketWatch": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><path d="M3 17l4-8 4 4 4-6 4 4" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 20h18" stroke={c} strokeWidth="1.5" strokeLinecap="round"/></svg>),
    "FinViz": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><rect x="4" y="14" width="3" height="6" rx="1" fill={c} opacity="0.5"/><rect x="10" y="9" width="3" height="11" rx="1" fill={c} opacity="0.75"/><rect x="17" y="4" width="3" height="16" rx="1" fill={c}/></svg>),
    "Yahoo Finance": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>),
    "Reddit": (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><circle cx="12" cy="12" r="9" stroke={c} strokeWidth="1.8"/><path d="M8 13.5c0 1.5 1.8 2.5 4 2.5s4-1 4-2.5" stroke={c} strokeWidth="1.8" strokeLinecap="round"/><circle cx="9" cy="10" r="1" fill={c}/><circle cx="15" cy="10" r="1" fill={c}/></svg>),
  };
  return map[label] || (<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><circle cx="12" cy="12" r="9" stroke={c} strokeWidth="1.8"/></svg>);
}

const FALLBACK_SIGNALS = [
  { label: "Benzinga",      value: "Neutral", color: "#eab308" },
  { label: "CNBC",          value: "Neutral", color: "#eab308" },
  { label: "MarketWatch",   value: "Neutral", color: "#eab308" },
  { label: "FinViz",        value: "Neutral", color: "#eab308" },
  { label: "Yahoo Finance", value: "Neutral", color: "#eab308" },
  { label: "Reddit",        value: "Neutral", color: "#eab308" },
];

function LiveClock() {
  const [now, setNow] = useState(new Date());
  const [tzIdx, setTzIdx] = useState(0);
  const [ddOpen, setDdOpen] = useState(false);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const sel = TIMEZONES[tzIdx];
  const open = isMarketOpen(sel, now);
  const timeStr = now.toLocaleTimeString("en-GB", { timeZone: sel.tz, hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const dateStr = now.toLocaleDateString("en-CA", { timeZone: sel.tz });

  return (
    <div style={{ marginTop: "14px", padding: "14px 16px", borderRadius: "12px", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "7px", marginBottom: "8px" }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: open ? "#22c55e" : "#ef4444", boxShadow: open ? "0 0 6px #22c55e" : "0 0 6px #ef4444", animation: open ? "pulse-dot 1.5s infinite" : "none" }} />
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: open ? "#22c55e" : "#ef4444" }}>{open ? "MARKET OPEN" : "MARKET CLOSED"}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <span style={{ fontSize: 22, fontWeight: 700, color: "#fff", fontVariantNumeric: "tabular-nums" }}>{timeStr}</span>
        <span style={{ fontSize: 11, color: "#64748b" }}>{dateStr}</span>
      </div>
      <div style={{ position: "relative", marginTop: "10px" }}>
        <button onClick={() => setDdOpen(o => !o)} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", padding: "6px 10px", color: "#94a3b8", fontSize: 12, cursor: "pointer" }}>
          <span>{sel.flag} {sel.label}</span>
          <span style={{ opacity: 0.5, fontSize: 10 }}>▾</span>
        </button>
        {ddOpen && (
          <div style={{ position: "absolute", bottom: "calc(100% + 4px)", left: 0, right: 0, background: "#0f172a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", zIndex: 200, maxHeight: "220px", overflowY: "auto" }}>
            {TIMEZONES.map((tz, i) => {
              const o = isMarketOpen(tz, now);
              return (
                <div key={tz.tz} onClick={() => { setTzIdx(i); setDdOpen(false); }} style={{ padding: "8px 12px", fontSize: 12, cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", background: i === tzIdx ? "rgba(59,130,246,0.15)" : "transparent", color: i === tzIdx ? "#fff" : "#94a3b8", borderBottom: "1px solid rgba(255,255,255,0.04)" }} onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"} onMouseLeave={e => e.currentTarget.style.background = i === tzIdx ? "rgba(59,130,246,0.15)" : "transparent"}>
                  <span>{tz.flag} {tz.label}</span>
                  {tz.open && <span style={{ fontSize: 10, fontWeight: 700, color: o ? "#22c55e" : "#ef4444" }}>{o ? "OPEN" : "CLOSED"}</span>}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function MarketMoodPanel({ signals, composite }) {
  const mood = composite?.value || "Neutral";
  const moodColor = composite?.color || "#eab308";
  const articles = composite?.total_articles ?? 0;
  const bullish = signals.filter(s => s.value === "Bullish").length;
  const bearish = signals.filter(s => s.value === "Bearish").length;
  const neutral = signals.filter(s => s.value === "Neutral").length;
  const barWidth = mood === "Bullish" ? "78%" : mood === "Bearish" ? "32%" : "52%";

  return (
    <div style={{ marginTop: "14px", padding: "16px", borderRadius: "14px", background: "linear-gradient(135deg, rgba(14,165,233,0.08), rgba(15,23,42,0.96))", border: "1px solid rgba(14,165,233,0.22)", boxShadow: "0 0 24px rgba(14,165,233,0.08)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ color: "#38bdf8", fontSize: 11, fontWeight: 900, letterSpacing: "0.08em" }}>MARKET MOOD</div>
          <div style={{ color: "#475569", fontSize: 10, marginTop: 3 }}>News & sentiment impact</div>
        </div>
        <div style={{ color: moodColor, fontSize: 16, fontWeight: 900 }}>{mood}</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 11 }}>
        <div style={{ background: "rgba(255,255,255,0.035)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: "9px" }}>
          <div style={{ color: "#f8fafc", fontWeight: 900, fontSize: 14 }}>{articles}</div>
          <div style={{ color: "#64748b", fontSize: 9 }}>Articles scanned</div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.035)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: "9px" }}>
          <div style={{ color: signals.length > 0 ? "#22c55e" : "#eab308", fontWeight: 900, fontSize: 14 }}>{signals.length > 0 ? "Live" : "Waiting"}</div>
          <div style={{ color: "#64748b", fontSize: 9 }}>Signal status</div>
        </div>
      </div>
      <div style={{ height: 5, borderRadius: 99, background: "rgba(255,255,255,0.07)", overflow: "hidden", marginBottom: 8 }}>
        <div style={{ height: "100%", width: barWidth, background: moodColor, borderRadius: 99, transition: "width 0.8s ease" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", color: "#64748b", fontSize: 10, marginBottom: 10 }}>
        <span>Bearish</span>
        <span style={{ color: moodColor, fontWeight: 800 }}>{mood}</span>
        <span>Bullish</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
        {[{ label: "Bull", value: bullish, color: "#22c55e" }, { label: "Neutral", value: neutral, color: "#eab308" }, { label: "Bear", value: bearish, color: "#ef4444" }].map(x => (
          <div key={x.label} style={{ background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 8, padding: "7px 6px", textAlign: "center" }}>
            <div style={{ color: x.color, fontWeight: 900, fontSize: 12 }}>{x.value}</div>
            <div style={{ color: "#475569", fontSize: 8 }}>{x.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Tab bar ──────────────────────────────────────────────────────── */
const TABS = [
  { id: "prediction", label: "Prediction" },
  { id: "backtest",   label: "Backtest" },
  { id: "analysis",  label: "Analysis" },
];

function TabBar({ active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "rgba(255,255,255,0.03)", borderRadius: 12, padding: 4, border: "1px solid rgba(255,255,255,0.06)", width: "fit-content" }}>
      {TABS.map(tab => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          style={{
            padding: "7px 20px",
            borderRadius: 9,
            fontSize: 12,
            fontWeight: active === tab.id ? 700 : 400,
            cursor: "pointer",
            border: "none",
            transition: "all 0.2s",
            background: active === tab.id ? "rgba(59,130,246,0.2)" : "transparent",
            color: active === tab.id ? "#93c5fd" : "#475569",
            boxShadow: active === tab.id ? "0 0 0 1px rgba(59,130,246,0.3)" : "none",
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/* ── Refresh button ───────────────────────────────────────────────── */
function DataFreshnessBar({ data, ticker, onRefresh, refreshing }) {
  if (!data?.history?.length) return null;
  const lastDate = data.history[data.history.length - 1]?.Date;
  const today = new Date().toLocaleDateString("en-CA");
  const isStale = lastDate && lastDate < today;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 14px", borderRadius: 10, background: isStale ? "rgba(234,179,8,0.08)" : "rgba(34,197,94,0.06)", border: `1px solid ${isStale ? "rgba(234,179,8,0.2)" : "rgba(34,197,94,0.15)"}`, fontSize: 11 }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: isStale ? "#eab308" : "#22c55e", boxShadow: `0 0 5px ${isStale ? "#eab308" : "#22c55e"}` }} />
      <span style={{ color: "#64748b" }}>
        Last price: <span style={{ color: isStale ? "#eab308" : "#22c55e", fontWeight: 600 }}>{lastDate}</span>
        {isStale && <span style={{ color: "#eab308" }}> — stale</span>}
      </span>
      <button
        onClick={onRefresh}
        disabled={refreshing}
        style={{ marginLeft: 4, padding: "3px 10px", borderRadius: 7, border: "1px solid rgba(59,130,246,0.3)", background: "rgba(59,130,246,0.1)", color: "#93c5fd", fontSize: 10, cursor: refreshing ? "not-allowed" : "pointer", fontWeight: 600, opacity: refreshing ? 0.5 : 1 }}
      >
        {refreshing ? "Refreshing…" : "↻ Refresh"}
      </button>
    </div>
  );
}

export default function App() {
  const [data, setData]               = useState(null);
  const [loading, setLoading]         = useState(false);
  const [signals, setSignals]         = useState([]);
  const [composite, setComposite]     = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing]   = useState(false);
  const [dataRefreshing, setDataRefreshing] = useState(false);
  const [ticker, setTicker]           = useState("AAPL");
  const [activeTab, setActiveTab]     = useState("prediction");
  const [backtestData, setBacktestData] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [shapData, setShapData]       = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setData(null);
    setModelDetails(null);
    setShapData(null);
    setBacktestData(null);

    try {
      const res = await axios.get(`${API}/predict/${ticker}`);
      setData({ ...res.data, ticker_used: ticker });
    } catch (err) {
      console.error("API ERROR:", err);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  const refreshPriceData = useCallback(async () => {
    setDataRefreshing(true);
    try {
      await axios.post(`${API}/refresh/${ticker}`);
      await fetchData();
    } catch (err) {
      console.error("Refresh error:", err);
    } finally {
      setDataRefreshing(false);
    }
  }, [ticker, fetchData]);

  const refreshSignals = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await axios.get(`${API}/signals/${ticker}`);
      setSignals(res.data.signals || []);
      setComposite(res.data.composite || null);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Signal refresh error:", err);
    } finally {
      setRefreshing(false);
    }
  }, [ticker]);

  // Fetch backtest when tab switches
  const fetchBacktest = useCallback(async () => {
    if (backtestData) return;
    setBacktestLoading(true);
    try {
      const res = await axios.get(`${API}/backtest/${ticker}`);
      setBacktestData(res.data);
    } catch (err) {
      console.error("Backtest error:", err);
      setBacktestData({ error: "Backtest failed" });
    } finally {
      setBacktestLoading(false);
    }
  }, [ticker, backtestData]);

  // Fetch SHAP when analysis tab opens
  const fetchShap = useCallback(async () => {
    if (shapData) return;
    setShapLoading(true);
    try {
      const res = await axios.get(`${API}/shap/${ticker}`);
      setShapData(res.data);
    } catch (err) {
      console.error("SHAP error:", err);
    } finally {
      setShapLoading(false);
    }
  }, [ticker, shapData]);

  useEffect(() => { fetchData(); }, [fetchData]);
  useEffect(() => { refreshSignals(); const id = setInterval(refreshSignals, 20 * 60 * 1000); return () => clearInterval(id); }, [refreshSignals]);

  useEffect(() => {
    if (activeTab === "backtest") fetchBacktest();
    if (activeTab === "analysis") fetchShap();
  }, [activeTab, fetchBacktest, fetchShap]);

  const displaySignals = signals.length > 0 ? signals : FALLBACK_SIGNALS;

  return (
    <div className="min-h-screen text-white bg-[#020617] p-6">
      <style>{`
        @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        .signals-scroll::-webkit-scrollbar { width: 4px; }
        .signals-scroll::-webkit-scrollbar-track { background: transparent; }
        .signals-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
      `}</style>

      <TopBarPro ticker={ticker} setTicker={setTicker} fetchData={fetchData} />

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 20, marginBottom: 4 }}>
        <TabBar active={activeTab} onChange={setActiveTab} />
        <DataFreshnessBar data={data} ticker={ticker} onRefresh={refreshPriceData} refreshing={dataRefreshing} />
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* ── LEFT COLUMN (9 cols) ────────────────────────────────── */}
        <div className="col-span-9 space-y-6">

          {/* PREDICTION TAB */}
          {activeTab === "prediction" && (
            <>
              {data ? (
                <ChartPro
                  history={data.history}
                  forecast={data.forecast}
                  confidence_upper={data.confidence_upper}
                  confidence_lower={data.confidence_lower}
                  forecast_horizon={data.forecast_horizon}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-[420px] rounded-xl bg-white/5 gap-3">
                  {loading ? (
                    <><div style={{ width: 36, height: 36, borderRadius: "50%", border: "3px solid rgba(59,130,246,0.15)", borderTop: "3px solid #3b82f6", animation: "spin 1s linear infinite" }} /><span className="text-blue-400 text-xs">Running models for {ticker}…</span></>
                  ) : (
                    <span className="text-gray-500 text-sm">Enter a ticker and click Load</span>
                  )}
                </div>
              )}

              {/* Risk metrics — shown below chart */}
              {data?.risk_metrics && <RiskPanel risk={data.risk_metrics} />}

              <div className="glass neon-border p-4">
                <ModelsPro data={data} onDetailsUpdate={setModelDetails} />
              </div>
            </>
          )}

          {/* BACKTEST TAB */}
          {activeTab === "backtest" && (
            <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 16, border: "1px solid rgba(255,255,255,0.06)", padding: 28 }}>
              <BacktestPanel data={backtestData} loading={backtestLoading} />
            </div>
          )}

          {/* ANALYSIS TAB */}
          {activeTab === "analysis" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <ShapPanel data={shapData} loading={shapLoading} />

              {/* Model weights explanation */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 14, padding: 20 }}>
                <div style={{ color: "#38bdf8", fontSize: 11, fontWeight: 800, letterSpacing: "0.08em", marginBottom: 16 }}>
                  ENSEMBLE ARCHITECTURE
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
                  {[
                    { model: "Random Forest", weight: 18, color: "#22c55e", desc: "300 trees, max depth 12" },
                    { model: "XGBoost", weight: 24, color: "#3b82f6", desc: "Gradient boosting" },
                    { model: "LightGBM", weight: 24, color: "#8b5cf6", desc: "Leaf-wise growth" },
                    { model: "Logistic Reg.", weight: 12, color: "#eab308", desc: "Linear baseline" },
                    { model: "LSTM", weight: 22, color: "#f87171", desc: "10-step sequence" },
                  ].map(m => (
                    <div key={m.model} style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${m.color}33`, borderRadius: 12, padding: 14, textAlign: "center" }}>
                      <div style={{ color: m.color, fontSize: 22, fontWeight: 900, lineHeight: 1 }}>{m.weight}%</div>
                      <div style={{ color: "#94a3b8", fontSize: 11, fontWeight: 600, margin: "6px 0 4px" }}>{m.model}</div>
                      <div style={{ color: "#334155", fontSize: 9 }}>{m.desc}</div>
                      <div style={{ height: 3, borderRadius: 99, background: m.color, marginTop: 10, opacity: 0.5 }} />
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 16, padding: 14, background: "rgba(59,130,246,0.05)", borderRadius: 10, border: "1px solid rgba(59,130,246,0.1)", fontSize: 11, color: "#475569", lineHeight: 1.7 }}>
                  Base weights are dynamically adjusted at inference time using: <span style={{ color: "#94a3b8" }}>40% historical accuracy</span> + <span style={{ color: "#94a3b8" }}>40% current prediction confidence</span> + <span style={{ color: "#94a3b8" }}>20% recent 10-prediction performance</span>. A model that is highly confident and historically accurate gets amplified weight; a poorly performing model is downweighted.
                </div>
              </div>

              {/* Feature engineering summary */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 14, padding: 20 }}>
                <div style={{ color: "#38bdf8", fontSize: 11, fontWeight: 800, letterSpacing: "0.08em", marginBottom: 14 }}>
                  FEATURE GROUPS (28 total)
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                  {[
                    { group: "Price & Volume", count: 5, features: "OHLCV", color: "#3b82f6" },
                    { group: "Trend Indicators", count: 6, features: "MA5, MA10, Momentum5, Return, Return_lag×2", color: "#22c55e" },
                    { group: "Oscillators", count: 3, features: "RSI14, MACD, MACD_signal", color: "#a78bfa" },
                    { group: "Volatility", count: 2, features: "Volatility5, BB_percent", color: "#f87171" },
                    { group: "Volume", count: 3, features: "Volume_Ratio, High_ratio, Low_ratio", color: "#eab308" },
                    { group: "Sentiment (NLP)", count: 9, features: "sent_mean/std/count, lags, 3d/7d/momentum, avg_reliability", color: "#fb923c" },
                  ].map(g => (
                    <div key={g.group} style={{ background: "rgba(255,255,255,0.02)", border: `1px solid ${g.color}22`, borderRadius: 10, padding: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ color: g.color, fontSize: 11, fontWeight: 700 }}>{g.group}</span>
                        <span style={{ background: `${g.color}22`, color: g.color, fontSize: 10, fontWeight: 700, padding: "1px 7px", borderRadius: 99 }}>{g.count}</span>
                      </div>
                      <div style={{ color: "#334155", fontSize: 9, lineHeight: 1.6 }}>{g.features}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN (3 cols) ───────────────────────────────── */}
        <div className="col-span-3">
          {/* Verdict card — leads the rail with the final decision + unique meta */}
          {data && <FinalSignalPro data={data} details={modelDetails} composite={composite} />}

          <div className="glass neon-border p-5 flex flex-col">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-cyan-400 font-semibold flex items-center gap-2 text-sm">
                <svg viewBox="0 0 24 24" fill="none" width="14" height="14"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill="#facc15" stroke="#facc15" strokeWidth="0.5"/></svg>
                Market Signals
              </h2>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: refreshing ? "#3b82f6" : "#22c55e", boxShadow: refreshing ? "0 0 6px #3b82f6" : "0 0 4px #22c55e", animation: "pulse-dot 1.5s infinite" }} />
            </div>

            {lastUpdated && (
              <div className="text-[10px] text-slate-600 mb-3 flex items-center gap-1">
                <svg viewBox="0 0 24 24" fill="none" width="10" height="10"><circle cx="12" cy="12" r="9" stroke="#475569" strokeWidth="2"/><path d="M12 7v5l3 3" stroke="#475569" strokeWidth="2" strokeLinecap="round"/></svg>
                {refreshing ? <span className="text-blue-400">Refreshing…</span> : `Updated ${lastUpdated.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`}
              </div>
            )}

            {composite && composite.total_articles > 0 && (
              <div className="mb-3 px-3 py-2 rounded-xl flex items-center justify-between" style={{ background: composite.color + "18", border: `1px solid ${composite.color}44` }}>
                <div>
                  <div className="text-xs text-slate-400">Overall Sentiment</div>
                  <div className="text-[10px] text-slate-600">{composite.total_articles} articles · last 48h</div>
                </div>
                <span className="text-sm font-bold" style={{ color: composite.color }}>{composite.value}</span>
              </div>
            )}

            <div className="signals-scroll overflow-y-auto space-y-2 pr-1" style={{ flex: 1, maxHeight: "280px" }}>
              {displaySignals.map(({ label, value, color, count, score }) => (
                <div key={label} className="flex items-center justify-between px-3 py-2 rounded-xl" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center justify-center rounded-lg" style={{ width: 30, height: 30, flexShrink: 0, background: (color || "#eab308") + "22" }}>
                      <SignalIcon label={label} color={color} />
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-slate-300 leading-tight">{label}</div>
                      <div className="text-[10px] text-slate-500">{count != null ? `${count} article${count !== 1 ? "s" : ""}${score != null ? `  ${score > 0 ? "+" : ""}${score}` : ""}` : value}</div>
                    </div>
                  </div>
                  <span className="text-xs font-bold px-2 py-1 rounded-full whitespace-nowrap" style={{ color, background: (color || "#eab308") + "22" }}>{value}</span>
                </div>
              ))}
              {signals.length === 0 && <div className="text-center text-slate-600 text-xs pt-4 pb-2">Waiting for scrapers…</div>}
            </div>

            <LiveClock />
            <MarketMoodPanel signals={signals} composite={composite} />
          </div>
        </div>
      </div>
    </div>
  );
}
