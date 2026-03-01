from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime



def four_regime_allocation_from_prediction(
    pred: dict,
    bull_calm: float = 1.0,
    bull_turb: float = 0.9,
    bear_calm: float = 0.6,
    bear_turb: float = 0.2
):
    # Extract signals safely
    bull = pred.get("Stable_Bull", 0) == 1
    bear = pred.get("Stable_Bear", 0) == 1
    low_vol = pred.get("Stable_Low_Vol", 0) == 1
    high_vol = pred.get("High_Volatility", 0) == 1

    regime_final = "Neutral"
    position = 0.5  # default neutral exposure

    # High volatility priority
    if high_vol and bear:
        regime_final = "HighVol_Bear"
        position = bear_turb

    elif high_vol and bull:
        regime_final = "HighVol_Bull"
        position = bull_turb

    # Low volatility cases
    elif low_vol and bear:
        regime_final = "LowVol_Bear"
        position = bear_calm

    elif low_vol and bull:
        regime_final = "LowVol_Bull"
        position = bull_calm

    # Neutral inside bear
    elif bear:
        regime_final = "Neutral_Bear"
        position = bear_calm

    return {
        "Regime_Final": regime_final,
        "Recommended_Position_%": round(position * 100, 2),
        "Exposure_Fraction": round(position, 3)
    }


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "artefacts", "market_regime_models.pkl")

with open(model_path, "rb") as f:
    saved_obj = pickle.load(f)

models = saved_obj["models"]

@app.route("/", methods=['GET','POST','PUT','DELETE','UPDATE'])
def ping():
    return {"Message":"it's working"}

@app.route("/predict", methods=['GET'])
def predict():

    features_path = os.path.join(BASE_DIR, "data", "features.csv")
    features = pd.read_csv(features_path, index_col=0, parse_dates=True)

    if features.empty:
        return {"error": "Not enough data to compute features"}

    X_latest = features.tail(1)

    predictions = {}

    for regime in models:
        model = models[regime]
        X_input = X_latest[model.feature_names_in_]
        predictions[regime] = int(model.predict(X_input)[0])

    # Convert regime prediction → capital allocation
    allocation = four_regime_allocation_from_prediction(predictions)

    return {
        "Raw_Model_Output": predictions,
        "Final_Decision": allocation
    }


@app.route("/history", methods=["GET"])
def history():

    df = pd.read_csv("data/daily_summary.csv", parse_dates=["Date"])
    df = df.sort_values("Date")

    # reconstruct regime
    df["Regime"] = np.where(
        df["Stocks_Above_MA20"] > df["Stocks_Below_MA20"],
        "Bull",
        "Bear"
    )

    return {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "regimes": df["Regime"].tolist()
    }


@app.route("/backtest", methods=["GET"])
def backtest():

    bull_calm = float(request.args.get("bull_calm", 1.0))
    bull_turb = float(request.args.get("bull_turb", 0.9))
    bear_calm = float(request.args.get("bear_calm", 0.6))
    bear_turb = float(request.args.get("bear_turb", 0.2))
    cost = float(request.args.get("cost", 0.0005))

    prices = pd.read_csv("data/index_price.csv", parse_dates=["Date"])
    prices = prices.sort_values("Date")

    returns = prices["Close"].pct_change().fillna(0)

    # Simple example allocation (replace with real regime mapping)
    position = np.where(returns > 0, bull_calm, bear_calm)

    strategy = (1 + position * returns).cumprod()
    bh = (1 + returns).cumprod()

    dd = strategy / strategy.cummax() - 1

    return {
        "dates": prices["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "strategy": strategy.tolist(),
        "buy_hold": bh.tolist(),
        "drawdown": dd.tolist()
    }

@app.route("/metrics", methods=["GET"])
def metrics():

    prices = pd.read_csv("data/index_price.csv")
    returns = prices["Close"].pct_change().fillna(0)

    strategy_returns = 0.8 * returns
    ann = np.sqrt(252)

    sharpe = strategy_returns.mean()/strategy_returns.std()*ann
    cagr = (1+strategy_returns).prod()**(252/len(strategy_returns)) - 1
    vol = strategy_returns.std()*ann
    max_dd = ((1+strategy_returns).cumprod()/
              (1+strategy_returns).cumprod().cummax()-1).min()

    return {
        "cagr": round(cagr*100,2),
        "sharpe": round(sharpe,2),
        "volatility": round(vol*100,2),
        "max_drawdown": round(max_dd*100,2)
    }


@app.route("/importance", methods=["GET"])
def importance():

    model = models["Stable_Bull"]  # example
    imp = dict(zip(model.feature_names_in_,
                   model.feature_importances_))

    return imp

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
