# ---------------------------------------------------------
# Align feature matrix with price data used for prediction
# ---------------------------------------------------------
# Ensure features and prices share the same timeline
common_index = features.index.intersection(prices.index)
X = features.loc[common_index]
close = prices.loc[common_index]

# ---------------------------------------------------------
# Initialize signal DataFrame
# ---------------------------------------------------------
# Stores regime predictions aligned with dates and prices
signals = pd.DataFrame(index=common_index)
signals["Close"] = close

# ---------------------------------------------------------
# Generate regime signals using trained models
# ---------------------------------------------------------
# Each model predicts whether its regime is active (0/1)
for regime_name, model in models.items():
    signals[regime_name] = model.predict(X)

signals.tail()

signals.to_csv("market_regime_signals.csv")

def four_regime_strategy_clean(
    signals: pd.DataFrame,
    bull_calm: float = 1.0,
    bull_turb: float = 0.9,
    bear_calm: float = 0.6,
    bear_turb: float = 0.2
) -> pd.DataFrame:


    # Copy input signals to avoid side effects
    out = signals.copy()

    # ---------------------------------------------------------
    # 1. Extract regime signal flags
    # ---------------------------------------------------------
    bull     = out['Stable_Bull'] == 1
    bear     = out['Stable_Bear'] == 1
    low_vol  = out['Stable_Low_Vol'] == 1
    high_vol = out['High_Volatility'] == 1

    # ---------------------------------------------------------
    # 2. Assign a single exclusive regime per day
    # Priority: High Vol → Low Vol → Neutral
    # ---------------------------------------------------------
    out["Regime_Final"] = "Neutral"

    # High volatility regimes take precedence
    out.loc[high_vol & bear, "Regime_Final"] = "HighVol_Bear"
    out.loc[high_vol & bull, "Regime_Final"] = "HighVol_Bull"

    # Low volatility regimes only if still unassigned
    unassigned = out["Regime_Final"] == "Neutral"
    out.loc[unassigned & low_vol & bear, "Regime_Final"] = "LowVol_Bear"
    out.loc[unassigned & low_vol & bull, "Regime_Final"] = "LowVol_Bull"

    # ---------------------------------------------------------
    # 3. Map regimes to portfolio exposure
    # ---------------------------------------------------------
    out['Position'] = 0.9  # default Neutral position (non-bear)

    out.loc[out["Regime_Final"] == "LowVol_Bull",  'Position'] = bull_calm
    out.loc[out["Regime_Final"] == "HighVol_Bull", 'Position'] = bull_turb
    out.loc[out["Regime_Final"] == "LowVol_Bear",  'Position'] = bear_calm
    out.loc[out["Regime_Final"] == "HighVol_Bear", 'Position'] = bear_turb

    # Cap exposure during Neutral days inside a Bear market
    out.loc[
        (out["Regime_Final"] == "Neutral") & (bear),
        "Position"
    ] = bear_calm

    # ---------------------------------------------------------
    # 4. STICKY POSITION LOGIC
    # ---------------------------------------------------------
    # If regime does not change, keep previous position
    out['Position_Sticky'] = out['Position']

    for i in range(1, len(out)):
        if out['Regime_Final'].iloc[i] == out['Regime_Final'].iloc[i-1]:
            out.at[out.index[i], 'Position_Sticky'] = out['Position_Sticky'].iloc[i-1]

    # ---------------------------------------------------------
    # 5. Smooth exposure & apply Execution Lag
    # ---------------------------------------------------------
    # Rolling average smooths small regime flips
    out['Position_Smooth'] = out['Position_Sticky'].rolling(5, min_periods=1).mean()
    # Shift by one day to avoid look-ahead bias
    out['Position_Exec']   = out['Position_Smooth'].shift(1).fillna(0.5)

    return out

def backtest_clean(df, price_col='Close', pos_col='Position_Exec',
                   capital=1_000_000, cost_rate=0.0005):

    # Work on a copy to avoid mutating input data
    out = df.copy()

    # Daily asset returns
    out['R'] = out[price_col].pct_change().fillna(0)

    # Position exposure(0 to 1 scale)
    out['Pos'] = out[pos_col].fillna(0)

    # Strategy P/L before costs
    out['Gross'] = out['Pos'] * out['R']

    # Transaction cost = position turnover * cost_rate
    turnover = out['Pos'].diff().abs().fillna(0)
    out['Cost'] = -turnover * cost_rate

    # Net strategy returns after costs
    out['Net'] = out['Gross'] + out['Cost']

    # Equity curves
    out['Equity'] = (1 + out['Net']).cumprod() * capital
    out['BH_Equity'] = (1 + out['R']).cumprod() * capital

    return out

def evaluate_clean(returns_df):
    # Work on a copy to preserve original data
    r = returns_df.copy()

    # Extract strategy and benchmark series
    strat = r['Equity']      # strategy equity curve
    bh = r['BH_Equity']      # buy-and-hold equity curve
    net = r['Net']          # strategy daily returns
    ann_factor = np.sqrt(252)

    # Number of years in backtest
    n_days = len(r)
    years = n_days / 252

    # Peak-to-trough drawdowns
    strat_drawdown = strat / strat.cummax() - 1
    bh_drawdown = bh / bh.cummax() - 1

    # Buy-and-hold Sharpe
    bh_sharpe = r['R'].mean() / r['R'].std() * ann_factor if r['R'].std() != 0 else np.nan

    # CAGR calculations
    strat_cagr = (strat.iloc[-1] / strat.iloc[0])**(1 / years) - 1
    bh_cagr = (bh.iloc[-1] / bh.iloc[0])**(1 / years) - 1

    # ---------------------------------------------------------
    # Summary performance metrics
    # ---------------------------------------------------------
    result = {
        'Total Return %': (strat.iloc[-1] / strat.iloc[0] - 1) * 100,
        'CAGR % (Strategy)': strat_cagr * 100,
        'BH Return %': (bh.iloc[-1] / bh.iloc[0] - 1) * 100,
        'CAGR % (BH)': bh_cagr * 100,
        'Max Drawdown % (Strategy)': strat_drawdown.min() * 100,
        'BH Max Drawdown %': bh_drawdown.min() * 100,
        'Sharpe': net.mean() / net.std() * ann_factor if net.std() != 0 else np.nan,
        'BH Sharpe': bh_sharpe,
        'Win Rate %': (net > 0).mean() * 100,
        'Avg Position %': r['Pos'].mean() * 100,
        'Annual Vol %': net.std() * ann_factor * 100
    }

    # ---------------------------------------------------------
    # Formatted console output
    # ---------------------------------------------------------
    print("=======================================")
    print("📊 Strategy Evaluation Metrics")
    print("=======================================")
    for k, v in result.items():
        print(f"{k}: {v:.2f}")

    return pd.DataFrame(result, index=['Value']).round(3), strat_drawdown, bh_drawdown

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_strategy_results(returns_df, strat_drawdown, bh_drawdown):
    r = returns_df.copy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "📈 Equity Curve: Strategy vs Buy & Hold",
            "📉 Drawdown Comparison",
            "🔍 Position / Exposure"
        ]
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=r.index, y=r['Equity'],
        mode='lines', name='Strategy Equity'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=r.index, y=r['BH_Equity'],
        mode='lines', name='Buy & Hold',
        line=dict(dash='dash')
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=strat_drawdown.index, y=strat_drawdown,
        mode='lines', name='Strategy DD'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bh_drawdown.index, y=bh_drawdown,
        mode='lines', name='BH DD',
        line=dict(dash='dash')
    ), row=2, col=1)

    # Position
    fig.add_trace(go.Scatter(
        x=r.index, y=r['Pos'],
        mode='lines', name='Position'
    ), row=3, col=1)

    fig.update_layout(
        height=900,
        title="📊 Strategy Tear Sheet (Interactive)",
        template="plotly_white",
        legend=dict(orientation="h", y=1.05)
    )

    fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1)

    fig.show()

print("TESTING ENHANCED FOUR REGIME STRATEGY...")
# ---------------------------------------------------------
# Run end-to-end test of the enhanced four-regime strategy
# ---------------------------------------------------------

# Convert regime signals into position sizing decisions
signals_regime_v2 = four_regime_strategy_clean(signals)

# Backtest strategy performance with transaction costs
results_regime_v2 = backtest_clean(signals_regime_v2)

# Compute performance and risk metrics
results, strat_dd, bh_dd = evaluate_clean(results_regime_v2)

# Visualize equity, drawdowns, and position behavior
plot_strategy_results(results_regime_v2, strat_dd, bh_dd)

# ---------------------------------------------------------
# Generate final regime classification
# ---------------------------------------------------------
# Apply regime-to-position logic and assign one regime per day
out = four_regime_strategy_clean(signals)

# ---------------------------------------------------------
# Inspect regime frequency distribution
# ---------------------------------------------------------
# Counts how often each final regime occurs
# Useful for sanity checks and exposure diagnostics
regime_counts = out["Regime_Final"].value_counts()
print(regime_counts)

# ---------------------------------------------------------
# Analyze signal vs neutral day distribution
# ---------------------------------------------------------
# Signal days = days with a clear, actionable regime
signal_days = (out["Regime_Final"] != "Neutral").sum()

# Neutral days = no strong or mixed regime signals
no_signal_days = (out["Regime_Final"] == "Neutral").sum()

total_days = len(out)

# ---------------------------------------------------------
# Print signal coverage statistics
# ---------------------------------------------------------
print(f"Total days: {total_days}")
print(f"Signal days: {signal_days} ({signal_days/total_days:.2%})")
print(f"No / Mixed signal days: {no_signal_days} ({no_signal_days/total_days:.2%})")

# ---------------------------------------------------------
# Detect conflicting regime signals
# ---------------------------------------------------------

# Trend conflict: Bull and Bear signals active simultaneously
bull = out['Stable_Bull'] == 1
bear = out['Stable_Bear'] == 1
low_vol = out['Stable_Low_Vol'] == 1
high_vol = out['High_Volatility'] == 1

# Volatility conflict: Low and High volatility signals overlap
conflict_trend = bull & bear
conflict_vol   = low_vol & high_vol

# Store conflict flags for diagnostics and sanity checks
out["Trend_Conflict"] = conflict_trend
out["Vol_Conflict"]   = conflict_vol

# ---------------------------------------------------------
# Diagnose why days are classified as Neutral
# ---------------------------------------------------------
# Neutral days may arise from:
# - Conflicting trend signals
# - Conflicting volatility signals
# - No strong signals at all
neutral = out["Regime_Final"] == "Neutral"

# Breakdown of Neutral days by conflict type
neutral_breakdown = pd.Series({
    "Trend conflict only": (neutral & conflict_trend & ~conflict_vol).sum(),
    "Vol conflict only":   (neutral & conflict_vol & ~conflict_trend).sum(),
    "Both conflicts":     (neutral & conflict_trend & conflict_vol).sum(),
    "No signals at all":  (neutral & ~conflict_trend & ~conflict_vol).sum()
})

print(neutral_breakdown)

# ---------------------------------------------------------
# Regime-wise exposure diagnostics
# ---------------------------------------------------------
# Compute average position levels for each final regime
# Helps verify that exposure mapping behaves as intended
out.groupby("Regime_Final")[[
    "Position",        # raw regime-based exposure
    "Position_Sticky", # exposure after whipsaw control
    "Position_Exec"    # final, lagged executable exposure
]].mean()

def two_regime_strategy(signals: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid mutating original signals
    out = signals.copy()

    # Default position
    # Fully invested unless risk conditions are triggered
    out['Position'] = 1.0

    # ---------------------------------------------------------
    # Crash-like risk condition
    # ---------------------------------------------------------
    # Severe risk-off when:
    # - Bear market signal is present
    # - Bear regime is stable (persistent)
    # - Volatility is elevated
    crash_like = (
        (out['Bear_Market'] == 1) &
        (out['Stable_Bear'] == 1) &
        (out['High_Volatility'] == 1)
    )

    # ---------------------------------------------------------
    # Mild risk-off condition
    # ---------------------------------------------------------
    # Reduce exposure during stable bear regimes
    mild_risk_off = (
        (out['Stable_Bear'] == 1)
    )

    # ---------------------------------------------------------
    # Position sizing rules
    # ---------------------------------------------------------
    # Moderate de-risking in bear markets
    out.loc[mild_risk_off, 'Position'] = 0.7
    # Aggressive de-risking during crash-like conditions
    out.loc[crash_like, 'Position']    = 0.3

    # ---------------------------------------------------------
    # Smooth exposure and apply execution lag
    # ---------------------------------------------------------
    # Short smoothing window to avoid abrupt position changes
    out['Position_Smooth'] = out['Position'].rolling(3, min_periods=1).mean()
    # Lag positions by one day to avoid look-ahead bias
    out['Position_Exec']   = out['Position_Smooth'].shift(1).fillna(1.0)

    return out

# ---------------------------------------------------------
# Run tail-hedge (crash protection) strategy
# ---------------------------------------------------------
# Convert regime signals into defensive position sizing
signal_r = two_regime_strategy(signals)

# Backtest tail-hedge strategy with transaction costs
results_tail_hedge = backtest_clean(signal_r)

# Evaluate performance and risk metrics
results1, strat_dd1, bh_dd1 = evaluate_clean(results_tail_hedge)

# Visualize equity curve, drawdowns, and exposure behavior
plot_strategy_results(results_tail_hedge, strat_dd1, bh_dd1)

# ---------------------------------------------------------
# Analyze exposure by bear-regime state
# ---------------------------------------------------------
# Group by raw Bear_Market and Stable_Bear signals
# to understand how position sizing behaves across:
# - Early / unstable bear phases
# - Confirmed (stable) bear regimes
out.groupby(
    ['Bear_Market', 'Stable_Bear']
)['Position_Exec'].mean()

# ---------------------------------------------------------
# Identify crash-like regime conditions
# ---------------------------------------------------------
# Crash-like = confirmed bear + high volatility + price below long-term trend
crash_like = (
    (out['Bear_Market'] == 1) &
    (out['Stable_Bear'] == 1) &
    (out['High_Volatility'] == 1)
)

# ---------------------------------------------------------
# Inspect position behavior immediately after crash ends
# ---------------------------------------------------------
# Select days where:
# - Yesterday was crash-like
# - Today is no longer crash-like
# Used to verify how quickly exposure recovers post-crash
out[['Position_Exec']].loc[
    (crash_like.shift(1) == 1) & (crash_like == 0)
].head(10)