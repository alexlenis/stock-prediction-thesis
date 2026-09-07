import React, { useEffect, useRef } from "react";

// Normalize data
function normalize(data) {
  return data.map(d => ({
    x: new Date(d.Date).getTime(),
    y: Number(d.Close),
  }));
}

export default function CanvasChart({
  history,
  forecast,
  confidence_upper,
  confidence_lower,
}) {
  const canvasRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = 400;

    const h = normalize(history);
    const f = normalize(forecast);
    const u = normalize(confidence_upper);
    const l = normalize(confidence_lower);

    const all = [...h, ...f];

    const minX = Math.min(...all.map(d => d.x));
    const maxX = Math.max(...all.map(d => d.x));
    const minY = Math.min(...l.map(d => d.y));
    const maxY = Math.max(...u.map(d => d.y));

    const scaleX = x => ((x - minX) / (maxX - minX)) * W;
    const scaleY = y => H - ((y - minY) / (maxY - minY)) * H;

    ctx.clearRect(0, 0, W, H);

    // ───────── GRID ─────────
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    for (let i = 0; i < 6; i++) {
      const y = (H / 6) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // ───────── HISTOGRAM BACK ─────────
    h.forEach((d, i) => {
      const x = scaleX(d.x);
      const y = scaleY(d.y);

      ctx.fillStyle = i % 2 === 0
        ? "rgba(59,130,246,0.35)"
        : "rgba(59,130,246,0.25)";

      ctx.fillRect(x - 3, y, 6, H - y);
    });

    // ───────── FUTURE ZONE (RED FADE) ─────────
    const lastX = scaleX(h[h.length - 1].x);

    const gradient = ctx.createLinearGradient(lastX, 0, W, 0);
    gradient.addColorStop(0, "rgba(239,68,68,0.05)");
    gradient.addColorStop(1, "rgba(239,68,68,0.25)");

    ctx.fillStyle = gradient;
    ctx.fillRect(lastX, 0, W - lastX, H);

    // ───────── CONFIDENCE BAND ─────────
    ctx.beginPath();

    u.forEach((d, i) => {
      const x = scaleX(d.x);
      const y = scaleY(d.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    l.slice().reverse().forEach(d => {
      ctx.lineTo(scaleX(d.x), scaleY(d.y));
    });

    ctx.closePath();

    const bandGradient = ctx.createLinearGradient(0, 0, 0, H);
    bandGradient.addColorStop(0, "rgba(239,68,68,0.35)");
    bandGradient.addColorStop(1, "rgba(239,68,68,0.05)");

    ctx.fillStyle = bandGradient;
    ctx.fill();

    // ───────── SMOOTH LINE FUNCTION ─────────
    function drawSmoothLine(data, color, width, dashed = false, glow = false) {
      ctx.beginPath();
      ctx.lineWidth = width;
      ctx.strokeStyle = color;

      if (dashed) ctx.setLineDash([6, 6]);
      else ctx.setLineDash([]);

      for (let i = 0; i < data.length - 1; i++) {
        const x1 = scaleX(data[i].x);
        const y1 = scaleY(data[i].y);
        const x2 = scaleX(data[i + 1].x);
        const y2 = scaleY(data[i + 1].y);

        const cx = (x1 + x2) / 2;

        if (i === 0) ctx.moveTo(x1, y1);

        ctx.bezierCurveTo(cx, y1, cx, y2, x2, y2);
      }

      if (glow) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
      }

      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // ───────── HISTORY LINE ─────────
    drawSmoothLine(h, "white", 2);

    // ───────── FORECAST (MERGED) ─────────
    const merged = [h[h.length - 1], ...f];

    drawSmoothLine(merged, "rgba(239,68,68,0.3)", 6, false, true); // glow
    drawSmoothLine(merged, "#ef4444", 2, true); // dashed

    // ───────── PRICE LABELS ─────────
    const lastPrice = merged[merged.length - 1];
    const y = scaleY(lastPrice.y);

    ctx.fillStyle = "#ef4444";
    ctx.fillRect(W - 70, y - 12, 60, 24);

    ctx.fillStyle = "white";
    ctx.font = "12px sans-serif";
    ctx.fillText(lastPrice.y.toFixed(2), W - 60, y + 4);

  }, [history, forecast, confidence_upper, confidence_lower]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: 400 }}
    />
  );
}