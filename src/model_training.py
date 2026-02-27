from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def fit_evaluate_timeseries_rf(X, y, train_frac=0.7, n_splits=3, rf_kwargs=None):
    """
    Fits and evaluates a Random Forest classifier on time-series data.
    """
    # Handle optional model hyperparameters
    rf_kwargs = rf_kwargs or {}

    # ---------------------------------------------------------
    # Train / test split (time-aware)
    # ---------------------------------------------------------
    # Use the first `train_frac` portion for training
    # Remaining data is held out as a final test set
    n = len(X)
    cut = int(n * train_frac)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    # ---------------------------------------------------------
    # TimeSeries cross-validation on training data
    # ---------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_acc = []

    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Check if there are at least two unique classes in the training split
        if y_tr.nunique() < 2:
            print(f"Skipping CV fold due to insufficient data for training (only one class present).")
            continue

        # Train model on current fold
        m = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_kwargs)
        m.fit(X_tr, y_tr)

        # Check if there are at least two unique classes in the validation split for evaluation
        if y_val.nunique() < 2:
             print(f"Skipping CV fold evaluation due to insufficient data for validation (only one class present).")
             continue

        cv_acc.append(accuracy_score(y_val, m.predict(X_val)))

    # Final fit on full train
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_kwargs)
    model.fit(X_train, y_train)

    # Simple test eval
    test_acc = accuracy_score(y_test, model.predict(X_test))

    report = {
        "cv_fold_accuracy": cv_acc,
        "cv_mean_accuracy": float(np.mean(cv_acc)) if cv_acc else np.nan,
        "test_accuracy": float(test_acc),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    return model, report

def train_all_regime_models(features, regimes, output_path="regime_models.pkl"):
  # Stores trained models and their evaluation reports
    models = {}
    reports = {}

    # Align features & regimes labels by date
    common_index = features.index.intersection(regimes.index)
    X = features.loc[common_index]  # input features
    Y = regimes.loc[common_index]   # regime labels

    # ---------------------------------------------------------
    # Train one binary model per regime
    # ---------------------------------------------------------
    for regime in Y.columns:
        print(f"\n📈 Training model for regime: {regime}")
        y = Y[regime] # target for current regime (0/1)

        # Time-series RF with walk-forward CV and class balancing
        model, report = fit_evaluate_timeseries_rf(
            X=X,
            y=y,
            train_frac=0.7,
            n_splits=3,
            rf_kwargs={"n_estimators":100, "max_depth":10, "class_weight":"balanced"}
        )

        # Store model and diagnostics
        models[regime] = model
        reports[regime] = report
        print(report)

    # Save everything in one pickle file
    with open(output_path, "wb") as f:
        pickle.dump({"models": models, "reports": reports}, f)

    print(f"\nAll regime models saved to '{output_path}'")
    return models, reports


# 1. Create regime target labels from price series
regimes = define_market_regimes(prices, lookback=20)

# Align features & regimes labeled by date
common_index = features.index.intersection(regimes.index)
X = features.loc[common_index]
y = regimes.loc[common_index]

# Drop any remaining NaNs after alignment to ensure clean data for training
# This is crucial because define_market_regimes's final shift(1) introduces NaNs.
combined_data = pd.concat([X, y], axis=1)
combined_data.dropna(inplace=True)
X = combined_data[X.columns]
y = combined_data[y.columns]

# Split data into training and test sets
train_frac = 0.7
cut = int(len(X) * train_frac)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]
regimes_clean = y_test.copy() # Rename y_test to regimes_clean for clarity in evaluation function

# 2. Train and save all models
models, reports = train_all_regime_models(X_train, y_train, output_path="market_regime_models.pkl")

# 3. Evaluate models on test set
acc_table = pd.DataFrame(columns=['Regime', 'Accuracy'])
for regime in regimes_clean.columns:
    if regime in models:
        model = models[regime]
        # Check if there are at least two unique classes in the test split for evaluation
        if y_test[regime].nunique() < 2:
            print(f"Skipping test evaluation for regime '{regime}' due to insufficient data (only one class present).")
            continue
        test_acc = accuracy_score(regimes_clean[regime], model.predict(X_test))
        acc_table.loc[len(acc_table)] = [regime, test_acc]
    else:
        print(f"Model for regime '{regime}' not found.")

print("\n📊 Test Set Evaluation:")
print(acc_table)

from sklearn.metrics import classification_report

print("\n" + "="*50)
print("PRECISION, RECALL, F1-SCORE FOR EACH REGIME")
print("="*50)

for regime_name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n📊 {regime_name}:")
    print(classification_report(y_test[regime_name], y_pred))

def feature_importance_table(models, X_train, round_decimals=4):
    """
    Returns a sorted feature importance table:
    Rows   -> Features (sorted by mean importance)
    Columns-> Regimes
    Values -> Feature Importance
    """
#Code to print a table that shows tree-based feature importance extracted from separate
#Random Forest models trained for different market regimes.
#Higher values indicate that the model relied more heavily on that feature when making predictions.

    tables = []

    for regime, model in models.items():
        tables.append(
            pd.Series(
                model.feature_importances_,
                index=X_train.columns,
                name=regime
            )
        )

    importance_table = pd.concat(tables, axis=1)

    # ---- Sort by overall importance ----
    importance_table["Mean_Importance"] = importance_table.mean(axis=1)
    importance_table = importance_table.sort_values(
        by="Mean_Importance",
        ascending=False
    )

    return importance_table.round(round_decimals)

feature_importance = feature_importance_table(models=models,X_train=X_train)
feature_importance