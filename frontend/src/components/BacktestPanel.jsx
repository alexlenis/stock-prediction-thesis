import React, { useEffect, useRef } from "react";
import { createChart, LineStyle } from "lightweight-charts";

/* ── Stat card ───────────────────────────────────────────────────── */
function StatCard({ label, value, sub, color = "#f8fafc", accent = "#3b82f6" }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: `1px solid ${accent}33`,
      borderRadius: 12,
      padding: "14px 16px",
      textAlign: "center",
    }}>
      <div style={{ color: accent, fontSize: 9, fontWeight: 800, letterSpacing: "0.1em", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 22, fontWeight: 900, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ color: "#475569", fontSize: 10, marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

/* ── Horizontal accuracy bar ─────────────────────────────────────── */
function AccBar({ label, value, best }) {
  const pct   = Math.min(Math.max(value, 0), 100);
  const color = pct >= 55 ? "#22c55e" : pct >= 50 ? "#eab308" : "#ef4444";
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: best === label ? "#fff" : "#94a3b8", fontWeight: best === label ? 700 : 400 }}>
          {label}{best === label ? " ★" : ""}
        </span>
        <span style={{ fontSize: 11, fontWeight: 700, color }}>{value}%</span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 99, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

/* ── Monthly heatmap ─────────────────────────────────────────────── */
function MonthlyHeatmap({ monthly }) {
  if (!monthly || Object.keys(monthly).length === 0) return null;
  const entries = Object.entries(monthly).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div>
      <div style={{ fontSize: 11, color: "#475569", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 10 }}>
        MONTHLY ACCURACY
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {entries.map(([month, acc]) => {
          const color = acc >= 60 ? "#22c55e" : acc >= 50 ? "#eab308" : "#ef4444";
          const bg    = acc >= 60 ? "rgba(34,197,94,0.15)" : acc >= 50 ? "rgba(234,179,8,0.15)" : "rgba(239,68,68,0.15)";
          return (
            <div key={month} title={`${month}: ${acc}%`} style={{
              padding: "5px 8px",
              borderRadius: 8,
              background: bg,
              border: `1px solid ${color}44`,
              textAlign: "center",
              minWidth: 58,
            }}>
              <div style={{ color: "#94a3b8", fontSize: 9, marginBottom: 2 }}>
                {month.slice(2)}
              </div>
              <div style={{ color, fontSize: 12, fontWeight: 800 }}>{acc}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Recent trades table ─────────────────────────────────────────── */
function TradesTable({ trades }) {
  if (!trades || trades.length === 0) return null;
  return (
    <div>
      <div style={{ fontSize: 11, color: "#475569", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 10 }}>
        RECENT TRADES
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              {["Date", "Signal", "5-day Return", "Trade P&L", "✓"].map(h => (
                <th key={h} style={{ padding: "6px 8px", color: "#475569", textAlign: "left", fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...trades].reverse().slice(0, 15).map((t, i) => {
              const sigColor = t.signal === "BUY" ? "#22c55e" : t.signal === "SELL" ? "#ef4444" : "#eab308";
              const pnlColor = t.pnl > 0 ? "#22c55e" : t.pnl < 0 ? "#ef4444" : "#94a3b8";
              return (
                <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                  <td style={{ padding: "5px 8px", color: "#64748b" }}>{t.date}</td>
                  <td style={{ padding: "5px 8px" }}>
                    <span style={{ color: sigColor, fontWeight: 700 }}>{t.signal}</span>
                  </td>
                  <td style={{ padding: "5px 8px", color: t.return > 0 ? "#22c55e" : "#ef4444" }}>
                    {t.return > 0 ? "+" : ""}{t.return}%
                  </td>
                  <td style={{ padding: "5px 8px", color: pnlColor, fontWeight: 700 }}>
                    {t.pnl > 0 ? "+" : ""}{t.pnl}%
                  </td>
                  <td style={{ padding: "5px 8px" }}>
                    <span style={{ color: t.correct ? "#22c55e" : "#ef4444" }}>
                      {t.correct ? "✓" : "✗"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Cumulative returns chart ────────────────────────────────────── */
function ReturnsChart({ dates, strategy, buyhold }) {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current || !dates?.length) return;

    const chart = createChart(ref.current, {
      width:  ref.current.clientWidth,
      height: 260,
      layout: { background: { color: "transparent" }, textColor: "#475569", fontSize: 11 },
      grid: {
        vertLines: { color: "rgba(59,130,246,0.04)" },
        horzLines: { color: "rgba(59,130,246,0.06)" },
      },
      rightPriceScale: { borderColor: "rgba(59,130,246,0.15)" },
      timeScale: { borderColor: "rgba(59,130,246,0.15)", timeVisible: false },
      crosshair: { mode: 1 },
    });

    const makeData = (vals) =>
      dates.map((d, i) => ({ time: d, value: parseFloat((vals[i] ?? 0).toFixed(2)) }))
        .filter(d => d.time && d.value != null);

    const stratLine = chart.addLineSeries({
      color: "#22c55e", lineWidth: 2,
      priceLineVisible: false, lastValueVisible: true,
      title: "Strategy",
    });
    stratLine.setData(makeData(strategy));

    const bhLine = chart.addLineSeries({
      color: "#64748b", lineWidth: 1.5,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: true,
      title: "Buy & Hold",
    });
    bhLine.setData(makeData(buyhold));

    // Zero baseline
    const zeroLine = chart.addLineSeries({
      color: "rgba(255,255,255,0.12)", lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false, lastValueVisible: false,
    });
    zeroLine.setData(dates.map(d => ({ time: d, value: 0 })));

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
    });
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.remove(); };
  }, [dates, strategy, buyhold]);

  return <div ref={ref} style={{ width: "100%" }} />;
}

/* ── Main export ─────────────────────────────────────────────────── */
export default function BacktestPanel({ data, loading }) {
  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 400, gap: 12, color: "#475569" }}>
        <div style={{ width: 28, height: 28, borderRadius: "50%", border: "3px solid rgba(59,130,246,0.15)", borderTop: "3px solid #3b82f6", animation: "spin 1s linear infinite" }} />
        <span style={{ fontSize: 13 }}>Running walk-forward backtest…</span>
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div style={{ padding: 32, textAlign: "center", color: "#475569" }}>
        {data?.error || "No backtest data available."}
      </div>
    );
  }

  const { summary, model_accuracy, monthly_accuracy, trades, dates, strategy_equity, buyhold_equity } = data;
  const bestModel = Object.entries(model_accuracy || {}).sort(([, a], [, b]) => b - a)[0]?.[0];
  const outColor  = (summary?.outperformance ?? 0) >= 0 ? "#22c55e" : "#ef4444";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 800, color: "#f8fafc" }}>Walk-Forward Backtest</div>
          <div style={{ fontSize: 11, color: "#475569", marginTop: 3 }}>{summary?.test_period} · {summary?.test_rows} trading days held out</div>
        </div>
        <div style={{ fontSize: 10, color: "#3b82f6", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: 8, padding: "4px 10px" }}>
          20% hold-out · unseen data
        </div>
      </div>

      {/* Summary stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <StatCard
          label="ENSEMBLE ACCURACY"
          value={`${model_accuracy?.Ensemble ?? "—"}%`}
          sub={`vs 33% random baseline`}
          accent="#3b82f6"
          color={parseFloat(model_accuracy?.Ensemble) >= 50 ? "#22c55e" : "#ef4444"}
        />
        <StatCard
          label="STRATEGY RETURN"
          value={`${summary?.strategy_return >= 0 ? "+" : ""}${summary?.strategy_return}%`}
          sub="on $10k initial capital"
          accent="#22c55e"
          color={summary?.strategy_return >= 0 ? "#22c55e" : "#ef4444"}
        />
        <StatCard
          label="vs BUY & HOLD"
          value={`${summary?.outperformance >= 0 ? "+" : ""}${summary?.outperformance}%`}
          sub={`B&H: ${summary?.buyhold_return >= 0 ? "+" : ""}${summary?.buyhold_return}%`}
          accent={outColor}
          color={outColor}
        />
        <StatCard
          label="SHARPE RATIO"
          value={summary?.sharpe_ratio ?? "—"}
          sub={`Max drawdown: ${summary?.max_drawdown}%`}
          accent="#a78bfa"
        />
      </div>

      {/* Cumulative returns chart */}
      <div style={{ background: "rgba(13,31,60,0.6)", borderRadius: 14, border: "1px solid rgba(59,130,246,0.15)", overflow: "hidden" }}>
        <div style={{ padding: "12px 16px 4px", display: "flex", gap: 20, borderBottom: "1px solid rgba(59,130,246,0.06)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 18, height: 3, background: "#22c55e", borderRadius: 2 }} />
            <span style={{ fontSize: 11, color: "#64748b" }}>Ensemble Strategy</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 18, height: 3, background: "#64748b", borderRadius: 2, opacity: 0.6 }} />
            <span style={{ fontSize: 11, color: "#64748b" }}>Buy & Hold</span>
          </div>
          <div style={{ marginLeft: "auto", fontSize: 10, color: "#475569" }}>Cumulative return (%)</div>
        </div>
        <ReturnsChart dates={dates} strategy={strategy_equity} buyhold={buyhold_equity} />
      </div>

      {/* Model accuracy + monthly heatmap side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: 20 }}>
        <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.06)", padding: 18 }}>
          <div style={{ fontSize: 11, color: "#475569", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 14 }}>
            MODEL ACCURACY (TEST SET)
          </div>
          {Object.entries(model_accuracy || {}).sort(([, a], [, b]) => b - a).map(([name, val]) => (
            <AccBar key={name} label={name} value={val} best={bestModel} />
          ))}
          <div style={{ fontSize: 10, color: "#334155", marginTop: 10 }}>
            ★ Best individual model · 3-class: SELL / HOLD / BUY · Random = 33%
          </div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.06)", padding: 18 }}>
          <MonthlyHeatmap monthly={monthly_accuracy} />
          <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <StatCard label="TOTAL TRADES" value={summary?.total_trades ?? 0} sub="BUY + SELL signals fired" accent="#94a3b8" />
            <StatCard label="BUY WIN RATE" value={`${summary?.buy_win_rate}%`} sub="BUY predicts ↑ correctly" accent="#22c55e" />
          </div>
        </div>
      </div>

      {/* Trades table */}
      <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.06)", padding: 18 }}>
        <TradesTable trades={trades} />
      </div>
    </div>
  );
}
