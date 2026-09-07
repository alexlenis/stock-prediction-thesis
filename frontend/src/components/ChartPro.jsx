import React, { useEffect, useRef } from "react";
import { createChart, LineStyle, CrosshairMode } from "lightweight-charts";

function clean(arr) {
  if (!arr || arr.length === 0) return [];
  const seen = new Set();
  return arr
    .filter((d) => d && d.Date && d.Close != null)
    .map((d) => ({ time: String(d.Date), value: parseFloat(Number(d.Close).toFixed(2)) }))
    .sort((a, b) => (a.time > b.time ? 1 : -1))
    .filter((d) => {
      if (seen.has(d.time)) return false;
      seen.add(d.time);
      return true;
    });
}

export default function ChartPro({ history, forecast, confidence_upper, confidence_lower, forecast_horizon }) {
  const containerRef = useRef();
  const tooltipRef   = useRef();

  useEffect(() => {
    if (!history?.length || !forecast?.length) return;

    const h = clean(history);
    const f = clean(forecast);
    const u = clean(confidence_upper);
    const l = clean(confidence_lower);
    if (!h.length || !f.length) return;

    const lastH    = h[h.length - 1];
    const fMerged  = [lastH, ...f];
    const uMerged  = u.length ? [lastH, ...u] : fMerged;
    const lMerged  = l.length ? [lastH, ...l] : fMerged;

    // ── Chart setup ──────────────────────────────────────────────────
    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: "transparent" },
        textColor: "#475569",
        fontSize: 11,
        fontFamily: "'Space Grotesk', sans-serif",
      },
      grid: {
        vertLines: { color: "rgba(59,130,246,0.04)" },
        horzLines: { color: "rgba(59,130,246,0.06)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: "rgba(59,130,246,0.5)",
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: "#1e3a5f",
        },
        horzLine: {
          color: "rgba(59,130,246,0.5)",
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: "#1e3a5f",
        },
      },
      rightPriceScale: {
        borderColor: "rgba(59,130,246,0.15)",
        textColor: "#475569",
        scaleMargins: { top: 0.12, bottom: 0.08 },
      },
      timeScale: {
        borderColor: "rgba(59,130,246,0.15)",
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      handleScroll: true,
      handleScale: true,
    });

    // ── 1. Confidence upper — very subtle ────────────────────────────
    const confUpper = chart.addAreaSeries({
      topColor:             "rgba(239,68,68,0.12)",
      bottomColor:          "rgba(239,68,68,0.0)",
      lineColor:            "rgba(239,68,68,0.25)",
      lineWidth:            1,
      lineStyle:            LineStyle.Solid,
      priceLineVisible:     false,
      lastValueVisible:     false,
      crosshairMarkerVisible: false,
    });
    confUpper.setData(uMerged);

    // ── 2. Confidence lower — border only ────────────────────────────
    const confLower = chart.addAreaSeries({
      topColor:             "rgba(239,68,68,0.0)",
      bottomColor:          "rgba(239,68,68,0.0)",
      lineColor:            "rgba(239,68,68,0.25)",
      lineWidth:            1,
      lineStyle:            LineStyle.Solid,
      priceLineVisible:     false,
      lastValueVisible:     false,
      crosshairMarkerVisible: false,
    });
    confLower.setData(lMerged);

    // ── 3. History — neon blue glow ──────────────────────────────────
    const histArea = chart.addAreaSeries({
      topColor:    "rgba(59,130,246,0.25)",
      bottomColor: "rgba(59,130,246,0.01)",
      lineColor:   "#3b82f6",
      lineWidth:   2,
      priceLineVisible:       false,
      lastValueVisible:       false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius:  5,
      crosshairMarkerBorderColor:      "#93c5fd",
      crosshairMarkerBackgroundColor:  "#1d4ed8",
    });
    histArea.setData(h);

    // ── 4. Forecast — neon red dashed ────────────────────────────────
    const forecastLine = chart.addLineSeries({
      color:       "#f87171",
      lineWidth:   2,
      lineStyle:   LineStyle.Dashed,
      priceLineVisible:       false,
      lastValueVisible:       true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius:  5,
      crosshairMarkerBorderColor:     "#fca5a5",
      crosshairMarkerBackgroundColor: "#dc2626",
    });
    forecastLine.setData(fMerged);

    // ── 5. Split point marker ────────────────────────────────────────
    forecastLine.setMarkers([{
      time:     lastH.time,
      position: "inBar",
      color:    "rgba(255,255,255,0.8)",
      shape:    "circle",
      size:     1,
    }]);

    // ── Tooltip ──────────────────────────────────────────────────────
    const tooltip = tooltipRef.current;

    chart.subscribeCrosshairMove((param) => {
      if (!param?.time || !param?.point) {
        tooltip.style.display = "none";
        return;
      }
      const hVal = param.seriesData.get(histArea);
      const fVal = param.seriesData.get(forecastLine);
      const val  = hVal || fVal;
      if (!val) { tooltip.style.display = "none"; return; }

      const price      = val.value ?? 0;
      const isForecast = !!fVal && !hVal;
      const dateStr    = typeof param.time === "string"
        ? param.time
        : new Date(param.time * 1000).toISOString().slice(0, 10);

      const dotColor  = isForecast ? "#f87171" : "#60a5fa";
      const valColor  = isForecast ? "#f87171" : "#fff";
      const labelText = isForecast ? "Forecast" : "Historical";

      tooltip.innerHTML = `
        <div style="font-size:10px;color:#64748b;margin-bottom:4px;
                    letter-spacing:0.05em;">${dateStr}</div>
        <div style="font-size:18px;font-weight:700;color:${valColor};
                    text-shadow:0 0 12px ${dotColor}88;">
          $${price.toFixed(2)}
        </div>
        <div style="font-size:10px;margin-top:4px;color:${dotColor};
                    display:flex;align-items:center;gap:4px;">
          <div style="width:6px;height:6px;border-radius:50%;
                      background:${dotColor};
                      box-shadow:0 0 6px ${dotColor};"></div>
          ${labelText}
        </div>
      `;
      tooltip.style.display = "block";

      const w    = containerRef.current.clientWidth;
      const ttW  = 130;
      const left = param.point.x + ttW + 16 > w
        ? param.point.x - ttW - 16
        : param.point.x + 16;
      tooltip.style.left = left + "px";
      tooltip.style.top  = Math.max(0, param.point.y - 40) + "px";
    });

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (containerRef.current)
        chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => { ro.disconnect(); chart.remove(); };

  }, [history, forecast, confidence_upper, confidence_lower]);

  return (
    <div style={{
      position:     "relative",
      width:        "100%",
      background:   "rgba(13,31,60,0.6)",
      backdropFilter: "blur(12px)",
      borderRadius: "16px",
      overflow:     "hidden",
      border:       "1px solid rgba(59,130,246,0.25)",
      boxShadow:    "0 0 0 1px rgba(59,130,246,0.05), 0 0 30px rgba(59,130,246,0.08)",
    }}>

      {/* Legend */}
      <div style={{
        display: "flex", gap: "20px",
        padding: "14px 18px 8px",
        alignItems: "center",
        borderBottom: "1px solid rgba(59,130,246,0.08)",
      }}>
        {/* Past */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: 22, height: 3,
            background: "#3b82f6",
            borderRadius: 2,
            boxShadow: "0 0 6px #3b82f688",
          }}/>
          <span style={{ fontSize: 11, color: "#64748b" }}>Past</span>
        </div>

        {/* Forecast */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: 22, height: 3, borderRadius: 2,
            background: "repeating-linear-gradient(90deg,#f87171 0,#f87171 4px,transparent 4px,transparent 8px)",
            boxShadow: "0 0 6px #f8717188",
          }}/>
          <span style={{ fontSize: 11, color: "#64748b" }}>
            {forecast_horizon ? `${forecast_horizon}-day Forecast` : "Forecast"}
          </span>
        </div>

        {/* Confidence */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: 22, height: 10, borderRadius: 3,
            background: "rgba(239,68,68,0.12)",
            border: "1px solid rgba(239,68,68,0.3)",
          }}/>
          <span style={{ fontSize: 11, color: "#64748b" }}>80% Confidence</span>
        </div>

        {/* Spacer + live tag */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "6px" }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%",
            background: "#22c55e",
            boxShadow: "0 0 6px #22c55e",
            animation: "pulse-dot 1.5s infinite",
          }}/>
          <span style={{ fontSize: 10, color: "#22c55e", fontWeight: 600, letterSpacing: "0.08em" }}>
            LIVE
          </span>
        </div>
      </div>

      {/* Chart canvas */}
      <div ref={containerRef} style={{ width: "100%" }} />

      {/* Tooltip */}
      <div ref={tooltipRef} style={{
        display:       "none",
        position:      "absolute",
        pointerEvents: "none",
        background:    "rgba(2,6,23,0.95)",
        border:        "1px solid rgba(59,130,246,0.3)",
        borderRadius:  "10px",
        padding:       "10px 14px",
        zIndex:        100,
        minWidth:      "130px",
        boxShadow:     "0 0 20px rgba(59,130,246,0.15), 0 4px 24px rgba(0,0,0,0.6)",
      }}/>
    </div>
  );
}