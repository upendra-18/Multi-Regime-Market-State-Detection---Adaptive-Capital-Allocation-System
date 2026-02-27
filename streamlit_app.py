# ==========================================================
# Multi-Regime Market State Detection Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="Multi-Regime Allocation System",
    layout="wide",
)

# ==========================================================
# Utility Functions
# ==========================================================

@st.cache_data
def load_features():
    return pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

@st.cache_data
def load_signals():
    return pd.read_csv("data/daily_summary.csv", index_col=0, parse_dates=True)

@st.cache_resource
def load_models():
    with open("artefacts/market_regime_models.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["models"]

# ==========================================================
# Allocation Logic (Single Day)
# ==========================================================

def allocation_engine(pred, bull_calm, bull_turb, bear_calm, bear_turb):

    bull = pred.get("Stable_Bull", 0) == 1
    bear = pred.get("Stable_Bear", 0) == 1
    low_vol = pred.get("Stable_Low_Vol", 0) == 1
    high_vol = pred.get("High_Volatility", 0) == 1

    regime = "Neutral"
    position = 0.5

    if high_vol and bear:
        regime = "HighVol_Bear"
        position = bear_turb
    elif high_vol and bull:
        regime = "HighVol_Bull"
        position = bull_turb
    elif low_vol and bear:
        regime = "LowVol_Bear"
        position = bear_calm
    elif low_vol and bull:
        regime = "LowVol_Bull"
        position = bull_calm
    elif bear:
        regime = "Neutral_Bear"
        position = bear_calm

    return regime, position

# ==========================================================
# Sidebar Controls
# ==========================================================

st.sidebar.title("Strategy Controls")

bull_calm = st.sidebar.slider("Bull Calm Exposure", 0.5, 1.5, 1.0, 0.05)
bull_turb = st.sidebar.slider("Bull Turbulent Exposure", 0.5, 1.5, 0.9, 0.05)
bear_calm = st.sidebar.slider("Bear Calm Exposure", 0.0, 1.0, 0.6, 0.05)
bear_turb = st.sidebar.slider("Bear Turbulent Exposure", 0.0, 1.0, 0.2, 0.05)

cost_rate = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.05) / 100

# ==========================================================
# Load Data
# ==========================================================

features = load_features()
signals = load_signals()
models = load_models()

X_latest = features.tail(1)

# ==========================================================
# Predict Current Regime
# ==========================================================

predictions = {}
for regime in models:
    model = models[regime]
    X_input = X_latest[model.feature_names_in_]
    predictions[regime] = int(model.predict(X_input)[0])

regime_final, exposure = allocation_engine(
    predictions,
    bull_calm,
    bull_turb,
    bear_calm,
    bear_turb
)

# ==========================================================
# Top Panel – Current Status
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Regime", regime_final)
col2.metric("Recommended Allocation", f"{round(exposure*100,2)}%")
col3.metric("Model Signals Active", sum(predictions.values()))
col4.metric("Last Data Update", features.index[-1].date())

# ==========================================================
# Allocation Gauge
# ==========================================================

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=exposure*100,
    title={'text': "Portfolio Exposure %"},
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
st.plotly_chart(gauge, use_container_width=True)

# ==========================================================
# Backtest Engine
# ==========================================================

def backtest(df, exposure_series, cost_rate):

    r = df["Close"].pct_change().fillna(0)

    pos = exposure_series.shift(1).fillna(0.5)
    gross = pos * r
    turnover = pos.diff().abs().fillna(0)
    cost = -turnover * cost_rate
    net = gross + cost

    equity = (1 + net).cumprod()
    bh_equity = (1 + r).cumprod()

    return equity, bh_equity, net

# Dummy exposure (historical neutral for demonstration)
exposure_series = pd.Series(0.8, index=features.index)

equity, bh_equity, net = backtest(signals, exposure_series, cost_rate)

# ==========================================================
# Equity & Drawdown
# ==========================================================

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy"), row=1, col=1)
fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="Buy & Hold", line=dict(dash="dash")), row=1, col=1)

dd = equity / equity.cummax() - 1
bh_dd = bh_equity / bh_equity.cummax() - 1

fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Strategy DD"), row=2, col=1)
fig.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="BH DD", line=dict(dash="dash")), row=2, col=1)

fig.update_layout(height=700, title="Strategy vs Buy & Hold")
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Performance Metrics
# ==========================================================

ann_factor = np.sqrt(252)

sharpe = net.mean() / net.std() * ann_factor if net.std() != 0 else np.nan
cagr = equity.iloc[-1] ** (252/len(equity)) - 1
max_dd = dd.min()

m1, m2, m3 = st.columns(3)
m1.metric("CAGR %", round(cagr*100,2))
m2.metric("Sharpe Ratio", round(sharpe,2))
m3.metric("Max Drawdown %", round(max_dd*100,2))

# ==========================================================
# Feature Importance
# ==========================================================

st.subheader("Feature Importance")

try:
    model = list(models.values())[0]
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=model.feature_names_in_
        ).sort_values(ascending=False).head(10)

        fig_imp = go.Figure(go.Bar(
            x=importance.values,
            y=importance.index,
            orientation="h"
        ))
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
except:
    st.write("Feature importance not available for this model.")

# ==========================================================
# Regime Distribution
# ==========================================================

st.subheader("Regime Signal Distribution")

regime_counts = pd.Series(predictions).value_counts()
st.bar_chart(regime_counts)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")
st.markdown(
    "Multi-Regime Market State Detection & Risk-Adjusted Capital Allocation System"
)