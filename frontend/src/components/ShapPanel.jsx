import React from "react";

function cleanName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

export default function ShapPanel({ data, loading }) {
  if (loading) return (
    <div style={{ padding: "24px", textAlign: "center", color: "#475569", fontSize: 12 }}>
      Computing SHAP values…
    </div>
  );
  if (!data || !data.features || data.features.length === 0) return null;

  const features = [...data.features].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap)), 0.001);

  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 14,
      padding: 20,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <div>
          <div style={{ color: "#a78bfa", fontSize: 11, fontWeight: 800, letterSpacing: "0.08em" }}>
            SHAP FEATURE CONTRIBUTIONS
          </div>
          <div style={{ color: "#334155", fontSize: 10, marginTop: 2 }}>
            {data.model} · {data.target_class}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 10 }}>
          <span style={{ color: "#22c55e" }}>▶ Pushes BUY</span>
          <span style={{ color: "#ef4444" }}>◀ Pushes SELL</span>
        </div>
      </div>

      {features.map((f, i) => {
        const isPos  = f.shap >= 0;
        const color  = isPos ? "#22c55e" : "#ef4444";
        const pct    = Math.abs(f.shap) / maxAbs * 100;
        return (
          <div key={i} style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>
                {cleanName(f.feature)}
              </span>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontSize: 10, color: "#475569" }}>
                  val={typeof f.value === "number" ? f.value.toFixed(3) : f.value}
                </span>
                <span style={{ fontSize: 11, fontWeight: 700, color, minWidth: 56, textAlign: "right" }}>
                  {isPos ? "+" : ""}{f.shap.toFixed(4)}
                </span>
              </div>
            </div>
            <div style={{ height: 5, background: "rgba(255,255,255,0.04)", borderRadius: 99, overflow: "hidden", display: "flex" }}>
              {isPos ? (
                <>
                  <div style={{ flex: 1 }} />
                  <div style={{ width: `${pct / 2}%`, background: color, borderRadius: "0 99px 99px 0" }} />
                  <div style={{ flex: 1 }} />
                </>
              ) : (
                <>
                  <div style={{ flex: 1 }} />
                  <div style={{ width: `${pct / 2}%`, background: color, borderRadius: "99px 0 0 99px", marginLeft: "auto" }} />
                  <div style={{ flex: 1 }} />
                </>
              )}
            </div>
          </div>
        );
      })}

      <div style={{ marginTop: 12, fontSize: 10, color: "#1e293b" }}>
        SHAP (SHapley Additive exPlanations): each bar shows how much that feature pushed the model toward BUY (+) or SELL (−) for today's prediction.
      </div>
    </div>
  );
}
