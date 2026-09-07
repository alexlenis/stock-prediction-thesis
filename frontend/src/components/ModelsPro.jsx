import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import axios from "axios";

const MODEL_META = {
  rf: {
    name: "Random Forest", short: "RF", color: "#60a5fa",
    desc: "Ensemble of decision trees", type: "Tree Ensemble",
    icon: (c) => (
      <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
        <path d="M12 3v4M8 5l2 2M16 5l-2 2" stroke={c} strokeWidth="1.8" strokeLinecap="round"/>
        <path d="M6 10c0 3 2 5 6 5s6-2 6-5" stroke={c} strokeWidth="1.8"/>
        <path d="M9 15v4M15 15v4M7 19h10" stroke={c} strokeWidth="1.8" strokeLinecap="round"/>
      </svg>
    ),
  },
  xgb: {
    name: "XGBoost", short: "XGB", color: "#a78bfa",
    desc: "Gradient boosting machine", type: "Gradient Boost",
    icon: (c) => (
      <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
        <path d="M12 3L3 8l9 5 9-5-9-5z" stroke={c} strokeWidth="1.8" strokeLinejoin="round"/>
        <path d="M3 16l9 5 9-5M3 12l9 5 9-5" stroke={c} strokeWidth="1.8" strokeLinejoin="round"/>
      </svg>
    ),
  },
  lgbm: {
    name: "LightGBM", short: "LGBM", color: "#fbbf24",
    desc: "Light gradient boosting", type: "Gradient Boost",
    icon: (c) => (
      <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill={c} stroke={c} strokeWidth="0.5"/>
      </svg>
    ),
  },
  logistic: {
    name: "Logistic", short: "LOG", color: "#34d399",
    desc: "Logistic regression", type: "Linear Model",
    icon: (c) => (
      <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
        <path d="M3 12c2-6 14-6 18 0" stroke={c} strokeWidth="1.8" strokeLinecap="round"/>
        <path d="M3 12c2 6 14 6 18 0" stroke={c} strokeWidth="1.8" strokeLinecap="round"/>
        <circle cx="12" cy="12" r="2" fill={c}/>
      </svg>
    ),
  },
  lstm: {
    name: "LSTM", short: "LSTM", color: "#f472b6",
    desc: "Long short-term memory", type: "Neural Network",
    icon: (c) => (
      <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
        <rect x="3"  y="8" width="4" height="8" rx="1" stroke={c} strokeWidth="1.8"/>
        <rect x="10" y="5" width="4" height="14" rx="1" stroke={c} strokeWidth="1.8"/>
        <rect x="17" y="9" width="4" height="7" rx="1" stroke={c} strokeWidth="1.8"/>
        <path d="M7 12h3M14 12h3" stroke={c} strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
};

const SIGNAL_CFG = {
  BUY:  { color: "#22c55e", bar: 82 },
  SELL: { color: "#ef4444", bar: 18 },
  HOLD: { color: "#eab308", bar: 50 },
};

function ConfidenceArc({ value, color, size = 72 }) {
  const r = size / 2 - 6;
  const circ = 2 * Math.PI * r;
  const fill = ((value || 0) / 100) * circ;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="rgba(255,255,255,0.06)"
        strokeWidth="5"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeDasharray={`${fill} ${circ}`}
        strokeDashoffset={circ * 0.25}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dasharray 0.8s ease" }}
      />
      <text
        x={size / 2}
        y={size / 2 + 5}
        textAnchor="middle"
        fill={color}
        fontSize="13"
        fontWeight="700"
        fontFamily="monospace"
      >
        {value != null ? `${Math.round(value)}%` : "—"}
      </text>
    </svg>
  );
}

function Last10Dots({ last10 }) {
  const items = last10?.length ? last10 : Array(10).fill(null);

  return (
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
      {items.map((d, i) => {
        if (!d) {
          return (
            <div
              key={i}
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "rgba(255,255,255,0.08)",
              }}
            />
          );
        }

        const signalColor =
          d.pred === "BUY" ? "#22c55e" :
          d.pred === "SELL" ? "#ef4444" :
          "#eab308";

        return (
          <div
            key={i}
            title={`Pred: ${d.pred} | Actual: ${d.actual} | ${d.correct ? "✓" : "✗"}`}
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: signalColor,
              opacity: d.correct ? 1 : 0.3,
              cursor: "default",
            }}
          />
        );
      })}
    </div>
  );
}

function TopFactors({ factors, color }) {
  if (!factors?.length) {
    return (
      <div style={{ fontSize: 12, color: "#334155", paddingTop: 8 }}>
        Not available for this model
      </div>
    );
  }

  const maxVal = Math.max(...factors.map((f) => f.value));

  return (
    <>
      {factors.map(({ name, value }) => {
        const barPct = maxVal > 0 ? (value / maxVal) * 100 : 0;
        const pct = (value * 100).toFixed(1);

        return (
          <div key={name} style={{ marginBottom: 10 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                color: "#94a3b8",
                marginBottom: 4,
              }}
            >
              <span>{name}</span>
              <span style={{ color }}>{pct}%</span>
            </div>

            <div
              style={{
                height: 4,
                borderRadius: 2,
                background: "rgba(255,255,255,0.06)",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${barPct}%`,
                  borderRadius: 2,
                  background: color,
                  transition: "width 0.8s ease",
                }}
              />
            </div>
          </div>
        );
      })}
    </>
  );
}

function ModelModal({ modelKey, details, trainingPeriod, onClose }) {
  const meta = MODEL_META[modelKey];
  const d = details;
  const cfg = SIGNAL_CFG[d?.signal] || SIGNAL_CFG.HOLD;

  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") onClose();
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const riskLabel =
    d?.accuracy != null
      ? d.accuracy >= 55
        ? "Low"
        : d.accuracy >= 40
        ? "Med"
        : "High"
      : "—";

  const riskColor =
    d?.accuracy != null
      ? d.accuracy >= 55
        ? "#22c55e"
        : d.accuracy >= 40
        ? "#eab308"
        : "#ef4444"
      : "#64748b";

  const modal = (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 99999,
        background: "rgba(0,0,0,0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#0a1628",
          border: `1px solid ${meta.color}44`,
          borderRadius: "20px",
          padding: "28px",
          width: "500px",
          maxWidth: "95vw",
          maxHeight: "90vh",
          overflowY: "auto",
          position: "relative",
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "8px",
            padding: "4px 10px",
            color: "#64748b",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          ✕
        </button>

        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 22 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 12,
              flexShrink: 0,
              background: meta.color + "18",
              border: `1px solid ${meta.color}44`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {meta.icon(meta.color)}
          </div>

          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#fff" }}>
              {meta.name}
            </div>
            <div style={{ fontSize: 12, color: "#475569" }}>
              {meta.desc}
            </div>
          </div>

          <div style={{ marginLeft: "auto", fontSize: 24, fontWeight: 800, color: cfg.color }}>
            {d?.signal || "—"} {d?.signal === "BUY" ? "↑" : d?.signal === "SELL" ? "↓" : "→"}
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4,1fr)",
            gap: 10,
            marginBottom: 22,
          }}
        >
          {[
            {
              label: "Accuracy",
              value: d?.accuracy != null ? `${d.accuracy}%` : "—",
            },
            {
              label: "Win Rate",
              value: d?.win_rate != null ? `${d.win_rate}%` : "—",
            },
            {
              label: "Confidence",
              value: d?.confidence != null ? `${Math.round(d.confidence)}%` : "—",
            },
            {
              label: "Risk Level",
              value: riskLabel,
              color: riskColor,
            },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 10,
                padding: "10px 8px",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 16, fontWeight: 700, color: color || "#f1f5f9" }}>
                {value}
              </div>
              <div style={{ fontSize: 10, color: "#475569", marginTop: 2 }}>
                {label}
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 6,
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 12,
              padding: "14px 18px",
              flexShrink: 0,
            }}
          >
            <ConfidenceArc value={d?.confidence} color={cfg.color} size={80} />
            <span style={{ fontSize: 10, color: "#475569" }}>
              Confidence
            </span>
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: 11,
                color: "#475569",
                marginBottom: 10,
                fontWeight: 600,
                letterSpacing: "0.05em",
              }}
            >
              TOP CONTRIBUTING FACTORS
            </div>

            <TopFactors factors={d?.top_factors} color={meta.color} />
          </div>
        </div>

        <div
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 12,
            padding: "14px 16px",
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "#475569",
              fontWeight: 600,
              marginBottom: 10,
              letterSpacing: "0.05em",
            }}
          >
            LAST 10 PREDICTIONS
          </div>

          <Last10Dots last10={d?.last10} />

          <div style={{ marginTop: 10, fontSize: 10, color: "#334155" }}>
            Bright = correct &nbsp;·&nbsp; Dim = incorrect
          </div>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 16,
            paddingTop: 14,
            borderTop: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          {[
            { label: "Model type", value: meta.type },
            { label: "Training period", value: trainingPeriod || "2019 – 2026" },
            { label: "Last updated", value: "Just now" },
          ].map(({ label, value }) => (
            <div key={label} style={{ fontSize: 11, color: "#334155" }}>
              {label}
              <div style={{ color: "#64748b", fontWeight: 600, marginTop: 2 }}>
                {value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return ReactDOM.createPortal(modal, document.body);
}

function ModelCard({ modelKey, signal, details, onClick }) {
  const meta = MODEL_META[modelKey];
  const cfg = SIGNAL_CFG[signal] || SIGNAL_CFG.HOLD;
  const barRef = useRef();

  useEffect(() => {
    if (!barRef.current) return;

    const t = setTimeout(() => {
      if (barRef.current) {
        barRef.current.style.width = cfg.bar + "%";
      }
    }, 150);

    return () => clearTimeout(t);
  }, [signal, cfg.bar]);

  return (
    <div
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: 0,
        background: "rgba(255,255,255,0.02)",
        border: `1px solid ${meta.color}22`,
        borderRadius: "14px",
        padding: "14px 12px",
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        position: "relative",
        overflow: "hidden",
        cursor: "pointer",
        transition: "border-color 0.2s, transform 0.15s",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = meta.color + "55";
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = meta.color + "22";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: "10%",
          right: "10%",
          height: "1px",
          background: `linear-gradient(90deg,transparent,${meta.color}55,transparent)`,
        }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 8,
            flexShrink: 0,
            background: meta.color + "18",
            border: `1px solid ${meta.color}33`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {meta.icon(meta.color)}
        </div>

        <span
          style={{
            fontSize: 10,
            color: "#64748b",
            fontWeight: 600,
            letterSpacing: "0.03em",
          }}
        >
          {meta.short}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span
          style={{
            fontSize: 20,
            fontWeight: 800,
            color: cfg.color,
            letterSpacing: "0.05em",
          }}
        >
          {signal || "—"}
        </span>

        <ConfidenceArc value={details?.confidence} color={cfg.color} size={56} />
      </div>

      <Last10Dots last10={details?.last10} />

      <div
        style={{
          height: 3,
          borderRadius: 2,
          background: "rgba(255,255,255,0.06)",
          overflow: "hidden",
        }}
      >
        <div
          ref={barRef}
          style={{
            height: "100%",
            width: "0%",
            borderRadius: 2,
            background: cfg.color,
            transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
          }}
        />
      </div>

      <div
        style={{
          fontSize: 9,
          color: "#1e293b",
          textAlign: "center",
          letterSpacing: "0.05em",
        }}
      >
        CLICK FOR DETAILS
      </div>
    </div>
  );
}

export default function ModelsPro({ data, onDetailsUpdate }) {
  const [detailsPayload, setDetailsPayload] = useState(null);
  const [modal, setModal] = useState(null);

  useEffect(() => {
    if (!data) return;

    const ticker = data.ticker_used || "AAPL";

    axios
      .get(`http://127.0.0.1:8000/details/${ticker}`)
      .then((res) => {
        setDetailsPayload(res.data || null);

        if (onDetailsUpdate) {
          onDetailsUpdate(res.data || null);
        }
      })
      .catch((err) => {
        console.error("Details error:", err);

        setDetailsPayload(null);

        if (onDetailsUpdate) {
          onDetailsUpdate(null);
        }
      });
  }, [data, onDetailsUpdate]);

  const details = detailsPayload?.models || null;

  const models = [
    { key: "rf", signal: data?.rf },
    { key: "xgb", signal: data?.xgb },
    { key: "lgbm", signal: data?.lgbm },
    { key: "logistic", signal: data?.logistic },
    { key: "lstm", signal: data?.lstm },
  ];

  return (
    <>
      <div style={{ display: "flex", gap: 10 }}>
        {models.map(({ key, signal }) => (
          <ModelCard
            key={key}
            modelKey={key}
            signal={signal}
            details={details?.[key]}
            onClick={() => setModal(key)}
          />
        ))}
      </div>

      {modal && details?.[modal] && (
        <ModelModal
          modelKey={modal}
          details={details[modal]}
          trainingPeriod={detailsPayload?.training_period}
          onClose={() => setModal(null)}
        />
      )}
    </>
  );
}