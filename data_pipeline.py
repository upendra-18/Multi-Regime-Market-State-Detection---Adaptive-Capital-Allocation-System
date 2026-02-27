import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os


# -----------------------------------
# Fetch historical data (4 months safe window)
# -----------------------------------

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


# -----------------------------------
# Daily Indicators
# -----------------------------------

def calculate_daily_indicators(data):
  #Computing basic daily indicators used for calculating Market Breadth
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()     #calculating MA20
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()     #calculating MA50
    data['Daily_Return'] = data['Close'].pct_change() * 100                   #percentage change in a day
    data['5Day_Return'] = (data['Close'] / data['Close'].shift(5) - 1) * 100  #percentage change in 5 days
    return data


# -----------------------------------
# Scan single stock
# -----------------------------------

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

# -----------------------------------
# Scan All Stocks
# -----------------------------------
def scan_all_stocks(symbols, start_date, end_date):
#Runs the historical scanner across a list of symbols and aggregates results.
  all_results = []
  for symbol in symbols:
    res = scan_single_stock_historical(symbol, start_date, end_date)
    all_results.extend(res)
    print(f"{symbol}: {len(res)} days processed")
  return pd.DataFrame(all_results)


# -----------------------------------
# Generate Daily Summary
# -----------------------------------
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

# -----------------------------------
# Save Detailed Results
# -----------------------------------

def save_detailed_results(df, filename="data.csv"):
    """
    Saves DataFrame inside the data/ folder.
    Overwrites file if it already exists.
    """

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    filepath = os.path.join("data", filename)

    # Overwrite mode explicitly set
    df.to_csv(filepath, index=True, mode="w")

    print(f"✅ Saved (replaced if existed): {filepath}")


# -----------------------------------
# Creating Regime Features ( using daily_summary.csv & nifty500_data["close"] )
# -----------------------------------


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



# --------------------------------------
# Downloading & Saving daily_summary.csv 
# --------------------------------------


if __name__ == "__main__":
    # Load symbols from file that containes names of stocks in NIFTY500 (downloaded from NSE website)
    df_symbols = pd.read_csv("data/ind_nifty500list.csv")[['Symbol']].copy()
    df_symbols['Yahoo_Symbol'] = df_symbols['Symbol'] + '.NS'
    symbols = df_symbols['Yahoo_Symbol'].tolist()

    #defines historical scan window
    # Yesterday
    end_date = datetime.today() - timedelta(days=1)

    # 3 months before yesterday
    start_date = end_date - relativedelta(months=9)

    # Convert to string format YYYY-MM-DD
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    # Scans across all stocks
    all_results_df = scan_all_stocks(symbols, start_date, end_date)
    if all_results_df.empty:
        raise ValueError("No stock data fetched. Pipeline aborted.")
    save_detailed_results(all_results_df, "all_stock_data.csv")

    # Generate and save market breadth summary
    daily_summary_df = generate_daily_summary(all_results_df)
    save_detailed_results(daily_summary_df, "daily_summary.csv")


    #--------------------------------------------------------------------------------------

    #loads the csv file created as data
    data = pd.read_csv("data/daily_summary.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")

    #downloads OHLCV data of NIFTY500 for the desired period
    nifty500_data = yf.download("^CRSLDX", start=start_date, end=end_date)
    if isinstance(nifty500_data.columns, pd.MultiIndex):
        nifty500_data.columns = nifty500_data.columns.get_level_values(0)

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
    
    #checking the issue of timezone
    print("Market breadth rows:", market_breadth.shape)
    print("Prices rows:", prices.shape)
    print("Common dates:", len(common_dates))

    #resolving the issue of timezone
    market_breadth.index = pd.to_datetime(market_breadth.index).tz_localize(None)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)


    # ---------------------------------------------------------
    # Generate regime-level predictive features
    # ---------------------------------------------------------
    # Transforms aligned market breadth and price data into lagged, look-ahead-safe features for regime modeling
    features = create_regime_features(market_breadth, prices)

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
    
    for col in clip_features:
        if col in features.columns:
            # Limit extreme outliers while preserving direction
            features[col] = features[col].clip(-3, 3)

    # ---------------------------------------------------------
    # Saving the final Features version in data folder
    # ---------------------------------------------------------

    save_detailed_results(features, "features.csv")
    
