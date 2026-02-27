import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

API_URL = "https://multi-regime-market-state-detection.onrender.com/predict"

st.set_page_config(
    page_title="Market Regime Allocation Dashboard",
    layout="wide"
)

st.title("Multi-Regime Market State Detection & Capital Allocation")

# ==========================================================
# Fetch API Data
# ==========================================================

def fetch_prediction():
    try:
        response = requests.get(API_URL, timeout=20)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

data = fetch_prediction()

if data is None:
    st.stop()

# ==========================================================
# Extract Data
# ==========================================================

raw = data.get("Raw_Model_Output", {})
decision = data.get("Final_Decision", {})

regime = decision.get("Regime_Final", "N/A")
exposure = decision.get("Exposure_Fraction", 0)
exposure_pct = decision.get("Recommended_Position_%", 0)

# ==========================================================
# Top Metrics
# ==========================================================

col1, col2, col3 = st.columns(3)

col1.metric("Current Regime", regime)
col2.metric("Recommended Allocation", f"{exposure_pct}%")
col3.metric("Signals Active", sum(raw.values()))

st.markdown("---")

# ==========================================================
# Exposure Gauge
# ==========================================================

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=exposure_pct,
    title={'text': "Portfolio Exposure (%)"},
    gauge={
        'axis': {'range': [0, 150]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 40], 'color': "#ffcccc"},
            {'range': [40, 80], 'color': "#fff2cc"},
            {'range': [80, 150], 'color': "#ccffcc"},
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Raw Model Output
# ==========================================================

st.subheader("Raw Regime Signals")
st.json(raw)

# ==========================================================
# System Info
# ==========================================================

st.markdown("---")
st.caption("Backend powered by FastAPI | Hosted on Render | Updated via scheduled cron job")