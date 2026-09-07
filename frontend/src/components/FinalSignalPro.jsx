import React, { useMemo, useState } from "react";
import ReactDOM from "react-dom";

const SIGNAL_CFG = {
  BUY: {
    color: "#22c55e",
    glow: "rgba(34,197,94,0.22)",
    border: "rgba(34,197,94,0.35)",
    icon: "🚀",
    label: "Bullish Momentum",
  },
  SELL: {
    color: "#ff5252",
    glow: "rgba(255,82,82,0.22)",
    border: "rgba(255,82,82,0.35)",
    icon: "📉",
    label: "Bearish Pressure",
  },
  HOLD: {
    color: "#facc15",
    glow: "rgba(250,204,21,0.22)",
    border: "rgba(250,204,21,0.35)",
    icon: "⏸️",
    label: "Neutral / Wait",
  },
};

const MODEL_META = {
  rf: { label: "RF", full: "Random Forest", color: "#60a5fa", icon: "♜" },
  xgb: { label: "XGB", full: "XGBoost", color: "#a78bfa", icon: "◆" },
  lgbm: { label: "LGBM", full: "LightGBM", color: "#facc15", icon: "⚡" },
  logistic: { label: "LOG", full: "Logistic Regression", color: "#34d399", icon: "⊙" },
  lstm: { label: "LSTM", full: "LSTM Neural Network", color: "#f472b6", icon: "▥" },
};

function getSignalColor(signal) {
  return SIGNAL_CFG[signal]?.color || "#94a3b8";
}

function countRecentCorrect(last10 = []) {
  if (!Array.isArray(last10) || last10.length === 0) return 0;
  return last10.filter((x) => x.correct).length;
}

function calcReliability(model) {
  if (!model) return 0;

  const accuracy = Number(model.accuracy ?? 0);
  const confidence = Number(model.confidence ?? 0);
  const last10 = Array.isArray(model.last10) ? model.last10 : [];

  const recent = last10.length
    ? (last10.filter((x) => x.correct).length / last10.length) * 100
    : 0;

  return accuracy * 0.4 + confidence * 0.4 + recent * 0.2;
}

function MetaModal({ decision, onClose }) {
  const c = SIGNAL_CFG[decision.signal] || SIGNAL_CFG.HOLD;

  const modal = (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 99999,
        background: "rgba(0,0,0,0.78)",
        backdropFilter: "blur(5px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 18,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 770,
          maxWidth: "96vw",
          maxHeight: "94vh",
          overflowY: "auto",
          background: "linear-gradient(135deg, #06111f 0%, #071426 100%)",
          border: "1px solid rgba(14,165,233,0.35)",
          borderRadius: 22,
          padding: 32,
          boxShadow: "0 0 60px rgba(14,165,233,0.12)",
          position: "relative",
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 20,
            right: 20,
            width: 42,
            height: 42,
            borderRadius: 12,
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.1)",
            color: "#64748b",
            cursor: "pointer",
            fontSize: 22,
          }}
        >
          ×
        </button>

        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 20,
            marginBottom: 28,
          }}
        >
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <div
              style={{
                width: 58,
                height: 58,
                borderRadius: 16,
                background: "rgba(14,165,233,0.12)",
                border: "1px solid rgba(14,165,233,0.45)",
                color: "#38bdf8",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 28,
                fontWeight: 900,
              }}
            >
              ✣
            </div>

            <div>
              <div
                style={{
                  color: "#f8fafc",
                  fontSize: 24,
                  fontWeight: 900,
                  marginBottom: 4,
                }}
              >
                Ensemble AI Decision Engine
              </div>
              <div style={{ color: "#64748b", fontSize: 15 }}>
                Hybrid Weighted Ensemble
              </div>
            </div>
          </div>

          <div style={{ textAlign: "right", paddingRight: 44 }}>
            <div
              style={{
                color: c.color,
                fontSize: 34,
                fontWeight: 900,
                letterSpacing: "0.06em",
              }}
            >
              {decision.signal}
            </div>
            <div style={{ color: "#64748b", fontSize: 14, marginTop: 4 }}>
              {decision.buy} BUY / {decision.hold} HOLD / {decision.sell} SELL
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 14,
            marginBottom: 26,
          }}
        >
          {[
            { label: "Meta Confidence", value: `${decision.confidence}%`, color: c.color },
            { label: "Agreement", value: `${decision.agreement}%`, color: "#f8fafc" },
            {
              label: "Risk",
              value: decision.risk,
              color:
                decision.risk === "Low"
                  ? "#22c55e"
                  : decision.risk === "High"
                  ? "#ff5252"
                  : "#facc15",
            },
            {
              label: "Bias",
              value: decision.bias,
              color:
                decision.bias === "Bullish"
                  ? "#22c55e"
                  : decision.bias === "Bearish"
                  ? "#ff5252"
                  : "#facc15",
            },
          ].map((item) => (
            <div
              key={item.label}
              style={{
                background: "rgba(255,255,255,0.035)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 14,
                padding: "18px 14px",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  color: item.color,
                  fontSize: 22,
                  fontWeight: 900,
                  marginBottom: 6,
                }}
              >
                {item.value}
              </div>
              <div style={{ color: "#475569", fontSize: 12 }}>
                {item.label}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            background: "rgba(14,165,233,0.08)",
            border: "1px solid rgba(14,165,233,0.22)",
            borderRadius: 16,
            padding: "18px 20px",
            marginBottom: 26,
          }}
        >
          <div
            style={{
              color: "#38bdf8",
              fontSize: 13,
              fontWeight: 900,
              letterSpacing: "0.12em",
              marginBottom: 14,
            }}
          >
            DECISION LOGIC
          </div>

          <div
            style={{
              color: "#94a3b8",
              fontSize: 15,
              lineHeight: 1.7,
            }}
          >
            The final signal is calculated from the five base models using a hybrid weighted method:
            <br />
            <b style={{ color: "#cbd5e1" }}>
              40% accuracy, 40% current confidence, 20% last-10 performance.
            </b>{" "}
            Models with stronger reliability have more influence on the final decision.
          </div>
        </div>

        <div
          style={{
            color: "#475569",
            fontSize: 13,
            fontWeight: 900,
            letterSpacing: "0.12em",
            marginBottom: 14,
          }}
        >
          MODEL CONTRIBUTIONS
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {decision.contributions.map((m) => {
            const meta = MODEL_META[m.key];

            return (
              <div
                key={m.key}
                style={{
                  display: "grid",
                  gridTemplateColumns: "42px 110px 90px 130px 110px 1fr",
                  alignItems: "center",
                  gap: 8,
                  background: "rgba(255,255,255,0.025)",
                  border: "1px solid rgba(255,255,255,0.065)",
                  borderRadius: 14,
                  padding: "12px 16px",
                }}
              >
                <div
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: 10,
                    background: `${meta.color}18`,
                    color: meta.color,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 900,
                    fontSize: 16,
                  }}
                >
                  {meta.icon}
                </div>

                <div
                  style={{
                    color: "#e2e8f0",
                    fontSize: 15,
                    fontWeight: 900,
                  }}
                >
                  {meta.label}
                </div>

                <div
                  style={{
                    color: getSignalColor(m.signal),
                    fontSize: 14,
                    fontWeight: 900,
                  }}
                >
                  {m.signal}
                </div>

                <div style={{ color: "#94a3b8", fontSize: 14 }}>
                  Conf {m.confidence}
                </div>

                <div style={{ color: "#94a3b8", fontSize: 14 }}>
                  Rel {m.reliability}%
                </div>

                <div
                  style={{
                    color: "#facc15",
                    fontSize: 14,
                    fontWeight: 900,
                  }}
                >
                  {m.recentCorrect}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  return ReactDOM.createPortal(modal, document.body);
}

export default function FinalSignalPro({ data, details, composite }) {
  const [open, setOpen] = useState(false);

  const decision = useMemo(() => {
    if (!data) return null;

    const models = details?.models || {};
    const ensemble = details?.ensemble || data?.ensemble || {};

    const modelKeys = ["rf", "xgb", "lgbm", "logistic", "lstm"];

    const votes = modelKeys
      .map((key) => ({
        key,
        signal: data[key] || models[key]?.signal,
        details: models[key],
      }))
      .filter((m) => ["BUY", "HOLD", "SELL"].includes(m.signal));

    const buy = votes.filter((m) => m.signal === "BUY").length;
    const hold = votes.filter((m) => m.signal === "HOLD").length;
    const sell = votes.filter((m) => m.signal === "SELL").length;

    const finalSignal = ensemble.signal || data.final || "HOLD";

    const maxVotes = Math.max(buy, hold, sell);
    const agreement = votes.length ? Math.round((maxVotes / votes.length) * 100) : 0;

    const avgConfidence =
      votes.length > 0
        ? votes.reduce((sum, m) => sum + Number(m.details?.confidence ?? 0), 0) / votes.length
        : 0;

    const confidence =
      ensemble.confidence != null
        ? Number(ensemble.confidence).toFixed(1)
        : avgConfidence > 0
        ? avgConfidence.toFixed(1)
        : "0.0";

    const risk =
      ensemble.risk ||
      (agreement >= 90 ? "Low" : agreement >= 60 ? "Medium" : "High");

    const sentimentBias = composite?.value;

    const bias =
      ensemble.bias ||
      (sentimentBias === "Bullish"
        ? "Bullish"
        : sentimentBias === "Bearish"
        ? "Bearish"
        : finalSignal === "BUY"
        ? "Bullish"
        : finalSignal === "SELL"
        ? "Bearish"
        : "Neutral");

    const contributions = modelKeys.map((key) => {
      const model = models[key] || {};
      const conf =
        model.confidence != null
          ? `${Number(model.confidence).toFixed(1)}%`
          : "—";

      return {
        key,
        signal: data[key] || model.signal || "—",
        confidence: conf,
        reliability: calcReliability(model).toFixed(1),
        recentCorrect: countRecentCorrect(model.last10),
      };
    });

    return {
      signal: finalSignal,
      confidence,
      agreement,
      risk,
      bias,
      buy,
      hold,
      sell,
      contributions,
    };
  }, [data, details, composite]);

  if (!data || !decision) return null;

  const c = SIGNAL_CFG[decision.signal] || SIGNAL_CFG.HOLD;

  const metaStats = [
    { label: "CONF", value: `${decision.confidence}%`, color: c.color },
    { label: "AGREE", value: `${decision.agreement}%`, color: "#f8fafc" },
    {
      label: "RISK",
      value: decision.risk,
      color: decision.risk === "Low" ? "#22c55e" : decision.risk === "High" ? "#ff5252" : "#facc15",
    },
    {
      label: "BIAS",
      value: decision.bias,
      color: decision.bias === "Bullish" ? "#22c55e" : decision.bias === "Bearish" ? "#ff5252" : "#facc15",
    },
  ];

  return (
    <>
      <div
        onClick={() => setOpen(true)}
        style={{
          position: "relative",
          overflow: "hidden",
          borderRadius: 16,
          border: `1px solid ${c.border}`,
          background: "linear-gradient(160deg, rgba(2,6,23,0.98) 0%, rgba(8,18,35,1) 100%)",
          padding: 18,
          cursor: "pointer",
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.04), 0 0 30px ${c.glow}`,
          marginBottom: 14,
        }}
      >
        {/* top accent strip in the signal colour */}
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: c.color, opacity: 0.6 }} />

        <div style={{ color: "#64748b", fontSize: 10, fontWeight: 800, letterSpacing: "0.16em", marginBottom: 14 }}>
          FINAL AI DECISION
        </div>

        {/* Signal hero — icon + label */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
          <div
            style={{
              width: 46, height: 46, borderRadius: 12, flexShrink: 0,
              background: `${c.color}18`, border: `1px solid ${c.border}`,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22,
            }}
          >
            {c.icon}
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ color: c.color, fontSize: 30, fontWeight: 900, letterSpacing: "0.06em", lineHeight: 1 }}>
              {decision.signal}
            </div>
            <div style={{ color: "#94a3b8", fontSize: 12, fontWeight: 700, marginTop: 4 }}>
              {c.label}
            </div>
          </div>
        </div>

        {/* Vote tally bar */}
        <div style={{ display: "flex", gap: 6, marginBottom: 14, fontSize: 10, fontWeight: 700 }}>
          <span style={{ flex: 1, textAlign: "center", padding: "4px 0", borderRadius: 6, background: "rgba(34,197,94,0.12)", color: "#22c55e" }}>{decision.buy} BUY</span>
          <span style={{ flex: 1, textAlign: "center", padding: "4px 0", borderRadius: 6, background: "rgba(250,204,21,0.12)", color: "#facc15" }}>{decision.hold} HOLD</span>
          <span style={{ flex: 1, textAlign: "center", padding: "4px 0", borderRadius: 6, background: "rgba(255,82,82,0.12)", color: "#ff5252" }}>{decision.sell} SELL</span>
        </div>

        {/* 2x2 meta grid — the data no other panel shows */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {metaStats.map((item) => (
            <div
              key={item.label}
              style={{
                background: "rgba(255,255,255,0.035)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 10,
                padding: "10px 12px",
              }}
            >
              <div style={{ color: item.color, fontSize: 16, fontWeight: 900, lineHeight: 1 }}>{item.value}</div>
              <div style={{ color: "#64748b", fontSize: 9, letterSpacing: "0.08em", marginTop: 5 }}>{item.label}</div>
            </div>
          ))}
        </div>

        <div style={{ color: "#334155", fontSize: 10, textAlign: "center", marginTop: 12, letterSpacing: "0.08em" }}>
          CLICK FOR FULL BREAKDOWN →
        </div>
      </div>

      {open && <MetaModal decision={decision} onClose={() => setOpen(false)} />}
    </>
  );
}