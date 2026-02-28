from flask import Flask
import pickle
import pandas as pd
import os

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
