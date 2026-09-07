import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
from datetime import datetime
import pandas_market_calendars as mcal
import streamlit.components.v1 as components

from predict_engine import predict_stock

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="AI Trading Dashboard",
    page_icon="📈"
)

# =========================
# SESSION STATE
# =========================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "result" not in st.session_state:
    st.session_state.result = None

# =========================
# REAL MARKET STATUS
# =========================
def get_market_status():
    nyse = mcal.get_calendar('NYSE')
    now_utc = pd.Timestamp.utcnow()

    schedule = nyse.schedule(
        start_date=now_utc.date(),
        end_date=now_utc.date()
    )

    if schedule.empty:
        return "🔴 MARKET CLOSED", "#EF4444"

    market_open = schedule.iloc[0]['market_open']
    market_close = schedule.iloc[0]['market_close']

    if market_open <= now_utc <= market_close:
        return "🟢 MARKET OPEN", "#22C55E"
    else:
        return "🔴 MARKET CLOSED", "#EF4444"

# =========================
# TOP BAR
# =========================
top1, top2 = st.columns([3, 2])

with top1:
    toggle = st.toggle("Theme", value=st.session_state.dark_mode)
    st.session_state.dark_mode = toggle

with top2:
    market, color = get_market_status()

    clock_color = "white" if st.session_state.dark_mode else "black"

    components.html(f"""
    <div style="text-align:center; padding:10px 0 20px 0;">

        <div id="clock" style="font-size:60px; font-weight:700; color:{clock_color};"></div>

        <div id="date" style="color:#9CA3AF; font-size:18px;"></div>

        <div style="margin-top:10px; color:{color}; font-weight:600;">
            {market}
        </div>

    </div>

    <script>
    function updateClock() {{
        const now = new Date();
        const time = now.toLocaleTimeString('en-GB', {{ hour12: false }});
        const date = now.toISOString().split('T')[0];
        document.getElementById("clock").innerHTML = time;
        document.getElementById("date").innerHTML = date;
    }}
    setInterval(updateClock, 1000);
    updateClock();
    </script>
    """, height=150)

    timezone = st.selectbox(
        "Timezone",
        pytz.all_timezones,
        index=pytz.all_timezones.index("Europe/Athens")
    )

# =========================
# THEME
# =========================
if st.session_state.dark_mode:
    plot_bg = "#0B1220"
    paper_bg = "#0B1220"
    font_color = "white"

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0B1220 0%, #0E1726 100%);
        color: #E5E7EB;
    }
    label {
        color: #E5E7EB !important;
    }
    .stTextInput input {
        background-color: #111827 !important;
        color: #E5E7EB !important;
        border-radius: 10px;
        border: 1px solid #1F2933;
    }
    .stButton button {
        background: linear-gradient(90deg, #2563EB, #1D4ED8);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    plot_bg = "white"
    paper_bg = "white"
    font_color = "black"

# =========================
# TITLE
# =========================
st.title("📈 AI Stock Prediction System")

# =========================
# INPUT
# =========================
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Predict"):
    st.session_state.predicted = True
    st.session_state.result = predict_stock(ticker)

# =========================
# RESULTS
# =========================
if st.session_state.predicted and st.session_state.result is not None:

    result = st.session_state.result

    st.subheader("Model Predictions")

    def color_signal(val):
        if val == "BUY":
            return "#22C55E"
        elif val == "SELL":
            return "#EF4444"
        else:
            return "#F59E0B"

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f"<h4>Random Forest</h4><h2 style='color:{color_signal(result['rf'])}'>{result['rf']}</h2>", unsafe_allow_html=True)
    col2.markdown(f"<h4>XGBoost</h4><h2 style='color:{color_signal(result['xgb'])}'>{result['xgb']}</h2>", unsafe_allow_html=True)
    col3.markdown(f"<h4>LightGBM</h4><h2 style='color:{color_signal(result['lgbm'])}'>{result['lgbm']}</h2>", unsafe_allow_html=True)
    col4.markdown(f"<h4>Logistic</h4><h2 style='color:{color_signal(result['logistic'])}'>{result['logistic']}</h2>", unsafe_allow_html=True)
    col5.markdown(f"<h4>LSTM</h4><h2 style='color:{color_signal(result['lstm'])}'>{result['lstm']}</h2>", unsafe_allow_html=True)

    st.markdown("## 🔥 Final Decision")
    st.success(result["final"])

    df = result["df"]
    future = result["future"]
    paths = result["paths"]

    st.markdown("## Price Forecast (Next 25 Days)")

    past_df = df.tail(50).copy()

    if "Date" in past_df.columns:
        past_dates = pd.to_datetime(past_df["Date"])
    else:
        past_dates = past_df.index

    past_prices = past_df["Close"]

    last_date = past_dates.iloc[-1]

    # include weekends
    future_dates = pd.date_range(start=last_date, periods=26)[1:]

    paths_array = np.array(paths)

    upper = np.percentile(paths_array, 90, axis=0)
    lower = np.percentile(paths_array, 10, axis=0)

    all_prices = list(past_prices) + list(future)
    all_paths_flat = paths_array.flatten()

    y_min = min(min(all_prices), np.min(all_paths_flat))
    y_max = max(max(all_prices), np.max(all_paths_flat))

    padding = (y_max - y_min) * 0.05
    y_min -= padding
    y_max += padding


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=past_dates,
        y=past_prices,
        mode="lines",
        name="Past",
        line=dict(color="#3b82f6", width=5)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        line=dict(color="rgba(255,0,0,0)"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        fill='tonexty',
        fillcolor='rgba(239,68,68,0.3)',
        line=dict(color="rgba(255,0,0,0)"),
        name="Confidence"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future,
        mode="lines",
        name="Forecast",
        line=dict(color="#ef4444", width=3, dash="dash")
    ))

    shapes = []
    all_dates = list(past_dates) + list(future_dates)

    for i in range(len(all_dates)):
        shapes.append(dict(
            type="line",
            x0=all_dates[i],
            x1=all_dates[i],
            y0=y_min,
            y1=all_prices[i] if i < len(all_prices) else y_min,
            line=dict(color="rgba(156,163,175,0.5)", width=1.5, dash="dot")
        ))

    fig.update_layout(
        shapes=shapes,
        height=550,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=font_color),

        margin=dict(t=80),

        xaxis=dict(type="date", showgrid=False),

        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)" if st.session_state.dark_mode else "rgba(0,0,0,0.05)"
        ),

        legend=dict(
            font=dict(size=14, color=font_color),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.15,
            x=0
        ),

        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
