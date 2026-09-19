"""
Supervised Regression Pipelines for Upfront Fare and Trip Duration Prediction
Supports Ridge, Random Forest, LightGBM, and XGBoost with evaluation metrics
(RMSE, MAE, R², MAPE), hyperparameter tuning, feature importance, and model serialization.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

def evaluate_predictions(y_true, y_pred):
    """Computes standard regression metrics: RMSE, MAE, R², and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid zero division in MAPE
    non_zero = y_true > 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    
    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2)
    }

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_name: str = "fare" # 'fare' or 'duration'
) -> dict:
    """
    Trains and compares multiple regression models across validation set.
    """
    print(f"\n" + "=" * 60)
    print(f"BENCHMARKING REGRESSION MODELS: {task_name.upper()} PREDICTION")
    print("=" * 60)
    
    models = {
        "Ridge_Baseline": Ridge(alpha=1.0),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.08,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predictions
        preds_val = model.predict(X_val)
        # Ensure non-negative predictions for fares and duration
        preds_val = np.maximum(preds_val, 0.0)
        
        metrics = evaluate_predictions(y_val, preds_val)
        metrics["Train_Time_Sec"] = round(train_time, 2)
        results[name] = metrics
        fitted_models[name] = model
        
        print(f"Done in {train_time:.1f}s | RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R²: {metrics['R2']}")

    # Determine best model by lowest RMSE
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\n>>> Best Performing Model for {task_name}: {best_model_name}")
    print(f"    Validation Metrics: {results[best_model_name]}")
    
    return {
        "results": results,
        "fitted_models": fitted_models,
        "best_model_name": best_model_name,
        "best_model": fitted_models[best_model_name]
    }

def save_model(model, filepath: str):
    """Serializes model to .pkl format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Successfully serialized model to {filepath}")

def load_model(filepath: str):
    """Loads serialized .pkl model."""
    return joblib.load(filepath)
