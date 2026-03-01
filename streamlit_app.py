# ==========================================================
# Multi-Regime Market State Detection Dashboard (Full Version)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

API_URL = "https://multi-regime-market-state-detection-adaptive-c-production.up.railway.app/predict"

st.set_page_config(page_title="Multi-Regime Allocation System", layout="wide")

# ==========================================================
# DATA LOADERS
# ==========================================================

@st.cache_data
def load_prices():
    return pd.read_csv("data/index_price.csv", index_col=0, parse_dates=True)

@st.cache_data
def load_signals():
    return pd.read_csv("data/daily_summary.csv", index_col=0, parse_dates=True)

prices = load_prices()["Close"]
signals = load_signals()

# ==========================================================
# FETCH LIVE API
# ==========================================================

def fetch_prediction():
    try:
        r = requests.get(API_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except:
        return None

api_data = fetch_prediction()

# ==========================================================
# SIDEBAR STRATEGY CONTROLS
# ==========================================================

st.sidebar.title("Strategy Controls")

bull_calm = st.sidebar.slider("Bull Calm Exposure", 0.5, 1.5, 1.0)
bull_turb = st.sidebar.slider("Bull Turbulent Exposure", 0.5, 1.5, 0.9)
bear_calm = st.sidebar.slider("Bear Calm Exposure", 0.0, 1.0, 0.6)
bear_turb = st.sidebar.slider("Bear Turbulent Exposure", 0.0, 1.0, 0.2)
cost_rate = st.sidebar.slider("Transaction Cost %", 0.0, 0.5, 0.05)/100

# ==========================================================
# LIVE STATUS PANEL
# ==========================================================

st.title("Multi-Regime Market State Detection & Capital Allocation System")

if api_data:

    final = api_data["Final_Decision"]
    raw = api_data["Raw_Model_Output"]

    regime = final["Regime_Final"]
    exposure = final["Exposure_Fraction"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Regime", regime)
    col2.metric("Recommended Allocation", f"{final['Recommended_Position_%']}%")

    risk = "High" if "HighVol" in regime else "Moderate" if "LowVol" in regime else "Neutral"
    col3.metric("Risk Level", risk)

    confidence = round(sum(raw.values())/len(raw)*100,1)
    col4.metric("Confidence Score", f"{confidence}%")

else:
    st.error("API not reachable")
    st.stop()

# ==========================================================
# ALLOCATION ENGINE (Historical)
# ==========================================================

def allocation_engine(df):
    out = df.copy()

    bull = out['Stocks_Above_MA20'] > out['Stocks_Below_MA20']
    bear = ~bull
    high_vol = out['Stocks_Up_4.5pct_Today'] + out['Stocks_Down_4.5pct_Today'] > 50
    low_vol = ~high_vol

    out["Regime_Final"] = "Neutral"

    out.loc[high_vol & bear, "Regime_Final"] = "HighVol_Bear"
    out.loc[high_vol & bull, "Regime_Final"] = "HighVol_Bull"
    out.loc[low_vol & bear, "Regime_Final"] = "LowVol_Bear"
    out.loc[low_vol & bull, "Regime_Final"] = "LowVol_Bull"

    out["Position"] = 0.9
    out.loc[out["Regime_Final"]=="LowVol_Bull","Position"] = bull_calm
    out.loc[out["Regime_Final"]=="HighVol_Bull","Position"] = bull_turb
    out.loc[out["Regime_Final"]=="LowVol_Bear","Position"] = bear_calm
    out.loc[out["Regime_Final"]=="HighVol_Bear","Position"] = bear_turb

    out["Position"] = out["Position"].shift(1).fillna(0.5)

    return out

alloc = allocation_engine(signals)

# ==========================================================
# BACKTEST
# ==========================================================

returns = prices.pct_change().fillna(0)
pos = alloc["Position"]
gross = pos * returns
turnover = pos.diff().abs().fillna(0)
cost = -turnover * cost_rate
net = gross + cost

equity = (1 + net).cumprod()
bh_equity = (1 + returns).cumprod()

# ==========================================================
# EQUITY + DRAWDOWN
# ==========================================================

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy"), row=1, col=1)
fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="Buy & Hold", line=dict(dash="dash")), row=1, col=1)

dd = equity / equity.cummax() - 1
bh_dd = bh_equity / bh_equity.cummax() - 1

fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Strategy DD"), row=2, col=1)
fig.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="BH DD", line=dict(dash="dash")), row=2, col=1)

fig.update_layout(height=800, title="Equity & Drawdown")
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

ann = np.sqrt(252)

cagr = equity.iloc[-1]**(252/len(equity)) - 1
sharpe = net.mean()/net.std()*ann if net.std()!=0 else 0
sortino = net.mean()/net[net<0].std()*ann if net[net<0].std()!=0 else 0
calmar = cagr/abs(dd.min()) if dd.min()!=0 else 0

m1,m2,m3,m4 = st.columns(4)
m1.metric("CAGR %", round(cagr*100,2))
m2.metric("Sharpe", round(sharpe,2))
m3.metric("Sortino", round(sortino,2))
m4.metric("Calmar", round(calmar,2))

# ==========================================================
# REGIME TIMELINE
# ==========================================================

st.subheader("Regime Timeline")

timeline = alloc["Regime_Final"].map({
    "HighVol_Bear": 1,
    "HighVol_Bull": 2,
    "LowVol_Bear": 3,
    "LowVol_Bull": 4
})

st.area_chart(timeline)

# ==========================================================
# ALLOCATION GAUGE
# ==========================================================

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=exposure*100,
    title={'text': "Current Allocation %"},
    gauge={'axis': {'range':[0,100]}}
))

st.plotly_chart(gauge, use_container_width=True)

# ==========================================================
# ARCHITECTURE SECTION
# ==========================================================

st.markdown("---")
st.markdown("""
### System Architecture

Data (Yahoo Finance)  
→ Feature Engineering  
→ ML Regime Models  
→ Allocation Logic  
→ Backtesting Engine  
→ FastAPI (Railway Deployment)  
→ Streamlit Dashboard  
→ GitHub Actions (Daily Data Pipeline)
""")

st.caption(f"Last Updated: {datetime.utcnow()} UTC")