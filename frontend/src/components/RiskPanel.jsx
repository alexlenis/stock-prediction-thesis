import React, { useMemo } from "react";

/* ── helpers ─────────────────────────────────────────────────────── */
const fmt  = (v, d = 2) => (v >= 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const fmtP = (p)        => "$" + Number(p).toFixed(2);
const pctChg = (p, base) => ((p - base) / base * 100).toFixed(1);

/* ── Return Distribution Histogram ──────────────────────────────── */
function Histogram({ histogram, var_95, es_95, probabilities }) {
  const bins = histogram || [];
  if (!bins.length) return null;

  const maxCount = Math.max(...bins.map(b => b.count), 1);
  const xs       = bins.map(b => b.x);
  const xMin     = xs[0];
  const xMax     = xs[xs.length - 1];
  const xRange   = xMax - xMin || 1;

  const VB_W = 500, VB_H = 130;
  const PAD  = { top: 22, right: 8, bottom: 24, left: 8 };
  const plotW = VB_W - PAD.left - PAD.right;
  const plotH = VB_H - PAD.top - PAD.bottom;
  const barW  = plotW / bins.length;

  const sx = x  => PAD.left + ((x - xMin) / xRange) * plotW;
  const sy = c  => PAD.top  + plotH - (c / maxCount) * plotH;
  const varX   = sx(var_95);
  const zeroX  = sx(0);
  const esX    = sx(Math.min(...xs));          // left edge for ES zone

  // Normal distribution overlay (for academic comparison)
  const mean = bins.reduce((s, b) => s + b.x * b.count, 0) /
               Math.max(bins.reduce((s, b) => s + b.count, 0), 1);
  const variance = bins.reduce((s, b) => s + b.count * (b.x - mean) ** 2, 0) /
                   Math.max(bins.reduce((s, b) => s + b.count, 0), 1);
  const sigma = Math.sqrt(variance) || 1;
  const normPath = bins.map((b, i) => {
    const density = Math.exp(-0.5 * ((b.x - mean) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
    const normCount = density * maxCount * sigma * Math.sqrt(2 * Math.PI) * (xRange / bins.length);
    const nx = PAD.left + (i / (bins.length - 1)) * plotW;
    const ny = sy(Math.min(normCount, maxCount));
    return `${i === 0 ? "M" : "L"} ${nx} ${ny}`;
  }).join(" ");

  const profit  = probabilities?.profit  ?? 50;
  const loss    = (100 - profit).toFixed(1);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: "0.07em" }}>
          RETURN DISTRIBUTION (500 PATHS)
        </span>
        <div style={{ display: "flex", gap: 12 }}>
          <span style={{ fontSize: 12, fontWeight: 800, color: "#22c55e" }}>
            P(profit) = {profit}%
          </span>
          <span style={{ fontSize: 12, fontWeight: 800, color: "#ef4444" }}>
            P(loss) = {loss}%
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} width="100%" style={{ overflow: "visible" }}>

        {/* ES zone (avg loss beyond VaR) */}
        <rect x={PAD.left} y={PAD.top} width={Math.max(0, varX - PAD.left)} height={plotH}
              fill="rgba(239,68,68,0.10)" />
        <text x={PAD.left + 4} y={PAD.top + 10} fill="#ef444499" fontSize={7} fontStyle="italic">
          ES zone
        </text>

        {/* Gain zone label */}
        <rect x={zeroX} y={PAD.top} width={plotW - (zeroX - PAD.left)} height={plotH}
              fill="rgba(34,197,94,0.04)" />

        {/* Bars */}
        {bins.map((bar, i) => {
          const bx = PAD.left + (i / bins.length) * plotW;
          const bh = (bar.count / maxCount) * plotH;
          const by = PAD.top + plotH - bh;
          const isVar = bar.x <= var_95;
          const color = isVar ? "#ef4444" : bar.is_loss ? "#f87171" : "#22c55e";
          const opacity = isVar ? 0.85 : 0.65;
          return (
            <rect key={i} x={bx + 0.5} y={by} width={barW - 1} height={bh}
                  fill={color} opacity={opacity} rx={1} />
          );
        })}

        {/* Normal distribution overlay */}
        <path d={normPath} fill="none" stroke="rgba(255,255,255,0.2)"
              strokeWidth={1.2} strokeDasharray="3,2" />

        {/* Zero line */}
        <line x1={zeroX} y1={PAD.top} x2={zeroX} y2={PAD.top + plotH}
              stroke="rgba(255,255,255,0.35)" strokeWidth={1} strokeDasharray="2,2" />
        <text x={zeroX} y={VB_H - 2} fill="#475569" fontSize={8} textAnchor="middle">0%</text>

        {/* VaR line */}
        <line x1={varX} y1={PAD.top - 4} x2={varX} y2={PAD.top + plotH}
              stroke="#ef4444" strokeWidth={1.5} strokeDasharray="3,2" />
        <text x={varX} y={PAD.top - 6} fill="#ef4444" fontSize={8} textAnchor="middle"
              fontWeight="bold">
          VaR
        </text>

        {/* ES label */}
        {es_95 > xMin && (
          <>
            <line x1={sx(es_95)} y1={PAD.top} x2={sx(es_95)} y2={PAD.top + plotH}
                  stroke="#f8717188" strokeWidth={1} strokeDasharray="2,3" />
            <text x={sx(es_95)} y={PAD.top - 6} fill="#f8717188" fontSize={7} textAnchor="middle">
              ES
            </text>
          </>
        )}

        {/* X-axis ticks */}
        {[-20, -15, -10, -5, 5, 10, 15, 20].map(v => {
          if (v < xMin - 1 || v > xMax + 1) return null;
          const tx = sx(v);
          return (
            <g key={v}>
              <line x1={tx} y1={PAD.top + plotH} x2={tx} y2={PAD.top + plotH + 3}
                    stroke="#1e293b" strokeWidth={1} />
              <text x={tx} y={VB_H - 2} fill="#1e293b" fontSize={7} textAnchor="middle">
                {v > 0 ? `+${v}%` : `${v}%`}
              </text>
            </g>
          );
        })}

        {/* Normal distribution legend */}
        <line x1={VB_W - 95} y1={10} x2={VB_W - 82} y2={10}
              stroke="rgba(255,255,255,0.25)" strokeWidth={1.2} strokeDasharray="3,2" />
        <text x={VB_W - 79} y={13} fill="#1e293b" fontSize={7}>Normal (ref.)</text>
      </svg>
    </div>
  );
}

/* ── Time evolution fan ──────────────────────────────────────────── */
function FanChart({ timeEvolution, currentPrice, steps = 18 }) {
  const te = timeEvolution;
  if (!te?.p50?.length) return null;

  const VB_W = 500, VB_H = 90;
  const PAD  = { top: 8, right: 42, bottom: 16, left: 8 };
  const plotW = VB_W - PAD.left - PAD.right;
  const plotH = VB_H - PAD.top - PAD.bottom;

  const allPrices = [...te.p5, ...te.p95, currentPrice];
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const pRange = maxP - minP || 1;
  const n = te.p50.length;

  const sx = i => PAD.left + (i / (n - 1)) * plotW;
  const sy = p => PAD.top  + (1 - (p - minP) / pRange) * plotH;

  const line = arr => arr.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(i)} ${sy(v)}`).join(" ");
  const area = (hi, lo) => {
    const top = hi.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(i)} ${sy(v)}`).join(" ");
    const bot = [...lo].reverse().map((v, i) => `L ${sx(lo.length - 1 - i)} ${sy(v)}`).join(" ");
    return `${top} ${bot} Z`;
  };

  const curY = sy(currentPrice);

  return (
    <div>
      <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 6 }}>
        UNCERTAINTY FAN  <span style={{ color: "#1e293b", fontWeight: 400 }}>· how confidence intervals widen over 18 trading days</span>
      </div>
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} width="100%">

        {/* P5–P95 outer band */}
        <path d={area(te.p95, te.p5)} fill="rgba(59,130,246,0.07)" />
        {/* P25–P75 inner band */}
        <path d={area(te.p75, te.p25)} fill="rgba(59,130,246,0.14)" />

        {/* P5 and P95 lines */}
        <path d={line(te.p5)}  fill="none" stroke="#ef444466" strokeWidth={1} />
        <path d={line(te.p95)} fill="none" stroke="#22c55e66" strokeWidth={1} />

        {/* Median line */}
        <path d={line(te.p50)} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1.5} />

        {/* Current price horizontal reference */}
        <line x1={PAD.left} y1={curY} x2={PAD.left + plotW} y2={curY}
              stroke="rgba(255,255,255,0.12)" strokeWidth={1} strokeDasharray="3,3" />

        {/* End labels */}
        {[
          { key: "p95", color: "#22c55e" },
          { key: "p75", color: "#86efac" },
          { key: "p50", color: "#ffffff" },
          { key: "p25", color: "#fca5a5" },
          { key: "p5",  color: "#ef4444" },
        ].map(({ key, color }) => (
          <text key={key}
                x={PAD.left + plotW + 3}
                y={sy(te[key][n - 1]) + 3}
                fill={color} fontSize={7}>
            ${te[key][n - 1].toFixed(0)}
          </text>
        ))}

        {/* Day labels */}
        {[1, 5, 10, 15, n - 1].map(i => {
          if (i >= n) return null;
          return (
            <text key={i} x={sx(i)} y={VB_H} fill="#1e293b" fontSize={7} textAnchor="middle">
              D{i + 1}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

/* ── Probability badges ──────────────────────────────────────────── */
function ProbGrid({ probabilities: p }) {
  if (!p) return null;
  const items = [
    { label: "P(gain > +10%)", value: p.gain_10, color: "#22c55e" },
    { label: "P(gain > +5%)",  value: p.gain_5,  color: "#86efac" },
    { label: "P(profit)",      value: p.profit,  color: "#ffffff", big: true },
    { label: "P(loss > -5%)",  value: p.loss_5,  color: "#fca5a5" },
    { label: "P(loss > -10%)", value: p.loss_10, color: "#ef4444" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
      {items.map(({ label, value, color, big }) => (
        <div key={label} style={{
          background: color + "11",
          border: `1px solid ${color}33`,
          borderRadius: 10,
          padding: "10px 8px",
          textAlign: "center",
        }}>
          <div style={{ color, fontSize: big ? 18 : 15, fontWeight: 900, lineHeight: 1 }}>
            {value}%
          </div>
          <div style={{ color: "#475569", fontSize: 9, marginTop: 5, lineHeight: 1.4 }}>{label}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Scenario price targets ──────────────────────────────────────── */
function ScenarioTargets({ price_targets }) {
  const pt = price_targets;
  if (!pt) return null;
  const current = pt.current;
  const rows = [
    { label: "Bull (P95)",   price: pt.p95, color: "#22c55e", icon: "▲" },
    { label: "P75",          price: pt.p75, color: "#86efac", icon: "↑" },
    { label: "Median (P50)", price: pt.p50, color: "#e2e8f0", icon: "→" },
    { label: "P25",          price: pt.p25, color: "#fca5a5", icon: "↓" },
    { label: "Bear (P5)",    price: pt.p5,  color: "#ef4444", icon: "▼" },
  ];
  return (
    <div>
      <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 10 }}>
        PRICE SCENARIOS — 18 DAYS
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rows.map(({ label, price, color, icon }) => {
          const chg = parseFloat(pctChg(price, current));
          return (
            <div key={label} style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "7px 10px", borderRadius: 9,
              background: color + "0d",
              border: `1px solid ${color}22`,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <span style={{ color, fontSize: 12 }}>{icon}</span>
                <span style={{ color: "#64748b", fontSize: 10 }}>{label}</span>
              </div>
              <div style={{ textAlign: "right" }}>
                <span style={{ color, fontWeight: 800, fontSize: 13 }}>{fmtP(price)}</span>
                <span style={{ color: chg >= 0 ? "#22c55e" : "#ef4444", fontSize: 10, marginLeft: 6 }}>
                  {chg >= 0 ? "+" : ""}{chg}%
                </span>
              </div>
            </div>
          );
        })}
        <div style={{ display: "flex", justifyContent: "center", marginTop: 4 }}>
          <span style={{ fontSize: 10, color: "#334155" }}>
            Current: <strong style={{ color: "#94a3b8" }}>${current}</strong>
          </span>
        </div>
      </div>
    </div>
  );
}

/* ── Distribution stats row ──────────────────────────────────────── */
function DistStats({ dist_stats, var_95, es_95 }) {
  const ds = dist_stats;
  if (!ds) return null;
  const items = [
    { label: "Mean Return",     value: fmt(ds.mean),    color: ds.mean  >= 0 ? "#22c55e" : "#ef4444" },
    { label: "Std Dev (σ)",     value: ds.std + "%",    color: "#94a3b8" },
    { label: "Skewness",        value: ds.skewness,     color: ds.skewness < 0 ? "#f87171" : "#86efac",
      note: ds.skewness < -0.2 ? "left tail" : ds.skewness > 0.2 ? "right tail" : "symmetric" },
    { label: "Excess Kurtosis", value: ds.kurtosis,     color: Math.abs(ds.kurtosis) > 1 ? "#f87171" : "#64748b",
      note: ds.kurtosis > 1 ? "fat tails ⚠" : "normal tails" },
    { label: "VaR (95%)",       value: fmt(var_95),     color: "#ef4444" },
    { label: "Exp. Shortfall",  value: fmt(es_95),      color: "#f87171" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 8 }}>
      {items.map(({ label, value, color, note }) => (
        <div key={label} style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.05)",
          borderRadius: 9, padding: "9px 10px", textAlign: "center",
        }}>
          <div style={{ color, fontWeight: 800, fontSize: 13 }}>{value}</div>
          <div style={{ color: "#334155", fontSize: 8, marginTop: 4 }}>{label}</div>
          {note && <div style={{ color: "#1e293b", fontSize: 7, marginTop: 2 }}>{note}</div>}
        </div>
      ))}
    </div>
  );
}

/* ── Main export ─────────────────────────────────────────────────── */
export default function RiskPanel({ risk }) {
  if (!risk || !risk.histogram) return null;

  const { var_95, es_95, histogram, probabilities, price_targets, dist_stats, time_evolution } = risk;

  return (
    <div style={{
      background: "rgba(7,15,35,0.85)",
      border: "1px solid rgba(239,68,68,0.18)",
      borderRadius: 16,
      padding: 22,
      display: "flex",
      flexDirection: "column",
      gap: 20,
      boxShadow: "0 0 40px rgba(239,68,68,0.05)",
    }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ color: "#f87171", fontSize: 12, fontWeight: 800, letterSpacing: "0.08em" }}>
          MONTE CARLO RISK ANALYTICS
        </div>
        <div style={{ fontSize: 10, color: "#334155" }}>
          500 simulated paths · GARCH(1,1) volatility · 18 trading-day horizon
        </div>
      </div>

      {/* Main 2-column: histogram + scenarios */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 220px", gap: 20, alignItems: "start" }}>
        <Histogram histogram={histogram} var_95={var_95} es_95={es_95} probabilities={probabilities} />
        <ScenarioTargets price_targets={price_targets} />
      </div>

      {/* Probability grid */}
      <ProbGrid probabilities={probabilities} />

      {/* Fan chart + dist stats side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
        <FanChart timeEvolution={time_evolution} currentPrice={price_targets?.current} />
        <DistStats dist_stats={dist_stats} var_95={var_95} es_95={es_95} />
      </div>

    </div>
  );
}
