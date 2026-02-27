# importing necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_historical_data(symbol, start_date, end_date):
    # Download historical data
    # threads=True speeds up download when fetching multiple symbols
    # group_by="symbol" ensures consistent structure across assets

  data = yf.download(symbol,start=start_date,end=end_date,threads=True,group_by="symbol")

    # yfinance returns MultiIndex columns when group_by="symbol" is used
    # For single-symbol downloads, flatten columns to standard OHLCV format

  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(1)
  return data

def calculate_daily_indicators(data):
  #Computing basic daily indicators used for calculating Market Breadth
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()     #calculating MA20
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()     #calculating MA50
    data['Daily_Return'] = data['Close'].pct_change() * 100                   #percentage change in a day
    data['5Day_Return'] = (data['Close'] / data['Close'].shift(5) - 1) * 100  #percentage change in 5 days
    return data

def scan_single_stock_historical(symbol, start_date, end_date):
    """
    Scans historical price data for a single stock and generates
    daily technical condition flags.Then it returns a list of daily signal dictionaries
    """
    data = fetch_historical_data(symbol, start_date, end_date)

    data = calculate_daily_indicators(data)
    results = []

    # Start from index 50 to ensure MA50 is fully formed
    for i in range(50, len(data)):
        day = data.iloc[i]

    # Build a daily signal snapshot
        results.append({
            'Symbol': symbol,
            'Date': day.name.strftime('%Y-%m-%d'),

            # Price & volume
            'Close': day['Close'],
            'Volume': day['Volume'],

            # Moving averages
            'MA20': day['MA20'],
            'MA50': day['MA50'],

            # Returns
            'Daily_Return': day['Daily_Return'],
            '5Day_Return': day['5Day_Return'],

            # Trend conditions
            'Above_MA20': day['Close'] > day['MA20'],
            'Below_MA20': day['Close'] < day['MA20'],
            'Above_MA50': day['Close'] > day['MA50'],
            'Below_MA50': day['Close'] < day['MA50'],

            # Large daily move flags (momentum / shock detection)
            'Up_4.5pct_Today': day['Daily_Return'] >= 4.5,
            'Down_4.5pct_Today': day['Daily_Return'] <= -4.5,

            # Large multi-day move flags (short-term momentum)
            'Up_20pct_5Days': day['5Day_Return'] >= 20,
            'Down_20pct_5Days': day['5Day_Return'] <= -20
        })
    return results

def scan_all_stocks(symbols, start_date, end_date):
#Runs the historical scanner across a list of symbols and aggregates results.
  all_results = []
  for symbol in symbols:
    res = scan_single_stock_historical(symbol, start_date, end_date)
    all_results.extend(res)
    print(f"{symbol}: {len(res)} days processed")
  return pd.DataFrame(all_results)

def generate_daily_summary(df):
   """
    Generates a daily market breadth summary from stock-level scan results.

    For each trading day, this function aggregates:
    - Total number of stocks scanned
    - Trend participation (Stocks above and below MA20 & MA50)
    - Momentum extremes (Price moved 5% in a day,Price moved 20% in past 5 days)

    """
   daily_summary = df.groupby('Date').agg({
        'Symbol': 'count',
        'Above_MA20': 'sum',
        'Below_MA20': 'sum',
        'Above_MA50': 'sum',
        'Below_MA50': 'sum',
        'Up_4.5pct_Today': 'sum',
        'Down_4.5pct_Today': 'sum',
        'Up_20pct_5Days': 'sum',
        'Down_20pct_5Days': 'sum'
    }).reset_index()

   daily_summary.columns = [
        'Date', 'Total_Stocks_Scanned',
        'Stocks_Above_MA20', 'Stocks_Below_MA20',
        'Stocks_Above_MA50', 'Stocks_Below_MA50',
        'Stocks_Up_4.5pct_Today', 'Stocks_Down_4.5pct_Today',
        'Stocks_Up_20pct_5Days', 'Stocks_Down_20pct_5Days'
    ]
   return daily_summary.sort_values('Date')

def save_detailed_results(df, filename="data.csv"):
  #saves as a csv file
    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    # Load symbols from file that containes names of stocks in NIFTY500 (downloaded from NSE website)
    df_symbols = pd.read_csv("ind_nifty500list.csv")[['Symbol']].copy()
    df_symbols['Yahoo_Symbol'] = df_symbols['Symbol'] + '.NS'
    symbols = df_symbols['Yahoo_Symbol'].tolist()

    #defines historical scan window
    start_date = "2015-01-01"
    end_date = "2025-11-30"

    # Scans across all stocks
    all_results_df = scan_all_stocks(symbols, start_date, end_date)
    save_detailed_results(all_results_df, "all_stock_data.csv")

    # Generate and save market breadth summary
    daily_summary_df = generate_daily_summary(all_results_df)
    save_detailed_results(daily_summary_df, "daily_summary.csv")

#loads the csv file created as data
data = pd.read_csv("daily_summary.csv",index_col=0,parse_dates=True)
data

#downloads the file in an excel format
data.to_excel("daily_stock_summary.xlsx", index=False)

#downloads OHLCV data of NIFTY500 for the desired period
nifty500_data = yf.download("^CRSLDX", start="2015-01-01", end="2025-12-30")
if isinstance(nifty500_data.columns, pd.MultiIndex):
    nifty500_data.columns = nifty500_data.columns.get_level_values(0)

nifty500_data

print(data.shape)
print(data.columns)
print(data.info())

print(nifty500_data.shape)
print(nifty500_data.columns)
print(nifty500_data.info())

def define_market_regimes(prices, lookback=20):
   """
   Defines market regimes(y variable) based on trend and volatility characteristics.

   """

   # ----------------------------------------
   # Compute returns and rolling volatility
   # ----------------------------------------
   returns = prices.pct_change()
   rolling_vol = returns.rolling(lookback).std()
   trend = prices.pct_change(lookback)

   regimes = pd.DataFrame(index=prices.index)

   # ---------------------------------------------------------
   # Normalize volatility and trend using rolling z-scores
   # ---------------------------------------------------------
   # Z-scores are computed relative to a longer-term (90-day) baseline
   # This allows regime classification to adapt over time

   vol_zscore = (rolling_vol - rolling_vol.rolling(90).mean()) / rolling_vol.rolling(90).std()
   trend_zscore = (trend - trend.rolling(90).mean()) / trend.rolling(90).std()


   # ---------------------------------------------------------
   # Raw regime classification
   # ---------------------------------------------------------
   # Volatility regimes
   regimes["High_Volatility"] = (vol_zscore > 0.5).astype(int)
   regimes["Low_Volatility"] = (vol_zscore < -0.5).astype(int)

   # Trend regimes
   regimes["Bull_Market"] = (trend_zscore > 0.5).astype(int)
   regimes["Bear_Market"] = (trend_zscore < -0.5).astype(int)

   # ---------------------------------------------------------
   # Regime persistence filters
   # ---------------------------------------------------------
   # Require regime conditions to persist for at least
   # 4 out of the last 5 days to reduce whipsaws
   regimes["Stable_Bull"] = (
       (regimes["Bull_Market"] == 1) &
       (regimes["Bull_Market"].rolling(5).sum() >= 4)  # 4 of last 5 days
   ).astype(int)

   regimes["Stable_Bear"] = (
       (regimes["Bear_Market"] == 1) &
       (regimes["Bear_Market"].rolling(5).sum() >= 4)
   ).astype(int)

   regimes["Stable_Low_Vol"] = (
       (regimes["Low_Volatility"] == 1) &
       (regimes["Low_Volatility"].rolling(5).sum() >= 4)
   ).astype(int)

   # Drop initial rows with insufficient data for rolling calculations
   regimes = regimes.dropna()

   return regimes

def create_regime_features(market_breadth, prices):
   """
    Constructs predictive regime-level features using market breadth
    and index price data.
    All features are shifted to avoid look-ahead bias.
    """
   print("\nSTEP 3: CREATING PREDICTIVE FEATURES...")

   features = pd.DataFrame(index=market_breadth.index)

   features['Advance_Decline_Ratio'] = (
      (market_breadth['Stocks_Up_4.5pct_Today'] + 1) /
      (market_breadth['Stocks_Down_4.5pct_Today'] + 1)
   )

   features['MA20_Strength'] = (
      market_breadth['Stocks_Above_MA20'] /
      market_breadth['Total_Stocks_Scanned']
   )

   features['Market_Momentum'] = (
      (market_breadth['Stocks_Above_MA20'] - market_breadth['Stocks_Below_MA20']) /
      market_breadth['Total_Stocks_Scanned']
   )

   features['Participation_Rate'] = (
      (market_breadth['Stocks_Up_4.5pct_Today'] + market_breadth['Stocks_Down_4.5pct_Today']) /
      market_breadth['Total_Stocks_Scanned']
   )

   features['Extreme_Move_Ratio'] = (
      (market_breadth['Stocks_Up_20pct_5Days'] + market_breadth['Stocks_Down_20pct_5Days']) /
      market_breadth['Total_Stocks_Scanned']
   )

   features['Extreme_Momentum_Bias'] = (
      (market_breadth['Stocks_Up_20pct_5Days'] - market_breadth['Stocks_Down_20pct_5Days']) /
      market_breadth['Total_Stocks_Scanned']
   )

   features['Extreme_Volatility_Indicator'] = (
      features['Extreme_Move_Ratio'].rolling(5).mean() /
      features['Extreme_Move_Ratio'].rolling(20).mean()
   )
   returns = prices.pct_change()
   features['Price_Volatility_Ratio'] = (
      (returns.rolling(5).std() / returns.rolling(20).std())
   )

   features['Trend_Strength'] = (
      (prices / prices.rolling(20).mean() - 1)
   )
   features = features.shift(1)

   lag_features = [
   'Advance_Decline_Ratio', 'MA20_Strength', 'Market_Momentum',
   'Participation_Rate', 'Extreme_Move_Ratio', 'Extreme_Momentum_Bias'
   ]
   for feat in lag_features:
     features[f'{feat}_lag1'] = features[feat].shift(1)

   features['Momentum_Change_5d'] = features['Market_Momentum'].diff(5)
   features['Breadth_Acceleration'] = features['MA20_Strength'].pct_change(3)
   features['Extreme_Momentum_Change'] = features['Extreme_Momentum_Bias'].diff(5)
   features['Extreme_Move_Acceleration'] = features['Extreme_Move_Ratio'].pct_change(3)

   # Remove rows with insufficient history
   features = features.dropna()

   print(f" Features created successfully: {features.shape[1]} features, {features.shape[0]} rows")

   return features

# ---------------------------------------------------------
# Align market breadth data with price series
# ---------------------------------------------------------
# Create a working copy of market breadth data
market_breadth = data.copy()
# Extract index price series (e.g., NIFTY 500 close)
prices = nifty500_data['Close']
# Find common dates between breadth data and price data
# This ensures features and targets are perfectly aligned
common_dates = market_breadth.index.intersection(prices.index)
# Subset both datasets to the shared date range
market_breadth = market_breadth.loc[common_dates]
prices = prices.loc[common_dates]

# ---------------------------------------------------------
# Generate regime-level predictive features
# ---------------------------------------------------------
# Transforms aligned market breadth and price data into lagged, look-ahead-safe features for regime modeling
features = create_regime_features(market_breadth, prices)

features_before = features.copy()

# ---------------------------------------------------------
# Feature normalization & outlier control
# ---------------------------------------------------------
# Log-transform skewed ratio-based features
log_features = [
    "Advance_Decline_Ratio",
    "Advance_Decline_Ratio_lag1",
    "Extreme_Move_Ratio",
    "Extreme_Move_Ratio_lag1",
    "Extreme_Momentum_Bias",
    "Extreme_Momentum_Bias_lag1"
]

for col in log_features:
    if col in features.columns:
        features[col] = np.log1p(features[col])

# ---------------------------------------------------------
# Clip acceleration features to control noise
# ---------------------------------------------------------
# Acceleration and change-based features are highly volatile
# Clipping prevents single-day shocks from dominating learning
clip_features = ["Breadth_Acceleration", "Extreme_Move_Acceleration"]
clip_features = ["Breadth_Acceleration", "Extreme_Move_Acceleration"]

for col in clip_features:
    if col in features.columns:
        # Limit extreme outliers while preserving direction
        features[col] = features[col].clip(-3, 3)


# Check total missing values per column
print(features.isna().sum())

print(features.describe())

# Feature correlation analysis
corr = features.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()

# Feature distribution inspection for skewness and outliers
features.hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.show()

changed_features = [
    col for col in features.columns
    if col in features_before.columns
    and not np.allclose(
        features_before[col].fillna(0),
        features[col].fillna(0)
    )
]

def plot_feature_changes(before, after, cols, bins=50):
    for col in cols:
        plt.figure(figsize=(8, 4))
        plt.hist(before[col].dropna(), bins=bins, alpha=0.6, label='Before')
        plt.hist(after[col].dropna(), bins=bins, alpha=0.6, label='After')
        plt.title(f"{col}: Before vs After")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_feature_changes(before, after, cols, bins=50):
    for col in cols:
        plt.figure(figsize=(8, 4))
        # Filter out non-finite values (NaN and inf) from 'before' data
        data_before = before[col].dropna()
        data_before = data_before[np.isfinite(data_before)]

        # Filter out non-finite values (NaN and inf) from 'after' data
        data_after = after[col].dropna()
        data_after = data_after[np.isfinite(data_after)]

        # Check if there's data to plot after filtering
        if not data_before.empty:
            plt.hist(data_before, bins=bins, alpha=0.6, label='Before')
        else:
            print(f"Warning: No finite data to plot for 'Before' in column '{col}'.")

        if not data_after.empty:
            plt.hist(data_after, bins=bins, alpha=0.6, label='After')
        else:
            print(f"Warning: No finite data to plot for 'After' in column '{col}'.")

        plt.title(f"{col}: Before vs After")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("Changed features:", changed_features)
plot_feature_changes(features_before, features, changed_features)