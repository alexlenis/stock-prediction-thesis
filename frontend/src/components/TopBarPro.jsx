import { useState, useEffect } from "react";

export default function TopBarPro({ ticker, setTicker, fetchData }) {
  const [input, setInput] = useState(ticker);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    setInput(ticker);
  }, [ticker]);

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const timeStr = time.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  const handleLoad = () => {
    const t = input.trim().toUpperCase();

    if (t) {
      setTicker(t);

      if (fetchData) {
        fetchData();
      }
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter") {
      handleLoad();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "18px 24px",
        background:
          "linear-gradient(135deg, rgba(7,15,28,0.96), rgba(4,10,20,0.98))",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(59,130,246,0.12)",
        borderRadius: "18px",
        boxShadow:
          "0 0 0 1px rgba(255,255,255,0.015), inset 0 1px 0 rgba(255,255,255,0.03)",
      }}
    >
      {/* LEFT */}
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div
          style={{
            width: 42,
            height: 42,
            borderRadius: 12,
            background:
              "linear-gradient(135deg, rgba(59,130,246,1), rgba(6,182,212,1))",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 22px rgba(14,165,233,0.22)",
            flexShrink: 0,
          }}
        >
          <svg viewBox="0 0 24 24" fill="none" width="18" height="18">
            <path
              d="M3 17l4-8 4 5 3-5 4 8"
              stroke="#fff"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <div>
          <div
            style={{
              fontSize: 16,
              fontWeight: 800,
              color: "#f8fafc",
              letterSpacing: "0.01em",
            }}
          >
            AI Trading Dashboard
          </div>

          <div
            style={{
              marginTop: 4,
              fontSize: 11,
              color: "#475569",
              letterSpacing: "0.05em",
              fontWeight: 600,
            }}
          >
            Multi-Model Stock Forecasting
          </div>
        </div>
      </div>

      {/* RIGHT */}
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {/* CLOCK */}
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 14,
            color: "#64748b",
            letterSpacing: "0.12em",
            fontVariantNumeric: "tabular-nums",
            paddingRight: 4,
          }}
        >
          {timeStr}
        </div>

        {/* SEARCH */}
        <div
          style={{
            width: 132,
            height: 46,
            borderRadius: 12,
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(59,130,246,0.18)",
            display: "flex",
            alignItems: "center",
            padding: "0 12px",
            gap: 8,
          }}
        >
          <svg viewBox="0 0 24 24" fill="none" width="14" height="14">
            <circle cx="11" cy="11" r="7" stroke="#475569" strokeWidth="2" />
            <path
              d="M16.5 16.5L21 21"
              stroke="#475569"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>

          <input
            value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyDown={handleKey}
            placeholder="AAPL"
            spellCheck={false}
            style={{
              width: "100%",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#f8fafc",
              fontSize: 13,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textAlign: "center",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          />
        </div>

        {/* BUTTON */}
        <button
          onClick={handleLoad}
          style={{
            height: 46,
            minWidth: 88,
            borderRadius: 12,
            border: "1px solid rgba(59,130,246,0.35)",
            background:
              "linear-gradient(135deg, rgba(59,130,246,0.95), rgba(37,99,235,0.95))",
            color: "#fff",
            fontSize: 13,
            fontWeight: 800,
            letterSpacing: "0.04em",
            cursor: "pointer",
            boxShadow: "0 0 18px rgba(59,130,246,0.18)",
            transition: "all 0.16s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow =
              "0 0 28px rgba(59,130,246,0.32)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow =
              "0 0 18px rgba(59,130,246,0.18)";
          }}
        >
          Load
        </button>
      </div>
    </div>
  );
}