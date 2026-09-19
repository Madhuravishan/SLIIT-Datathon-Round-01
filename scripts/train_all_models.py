"""
Master Model Training & Benchmarking Pipeline (Tracks 2 & 3)
Trains and benchmarks multiple regression models for Upfront Fare Prediction (2.1)
and On-Time Arrival Duration Estimation (2.2), plus Fleet Dispatcher Forecaster (3.1).
Exports serialized .pkl models and comprehensive evaluation metrics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import engineer_features, load_zone_lookup
from src.supervised_models import train_and_evaluate_models, evaluate_predictions, save_model
from src.demand_forecaster import (
    build_hourly_demand_series,
    create_time_series_features,
    train_demand_forecast_model,
    TOP_ZONES
)

DATA_SPLITS_DIR = "data_splits"
MODELS_DIR = "models"
METRICS_JSON = os.path.join(MODELS_DIR, "benchmark_metrics.json")

def run_training_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("STARTING FULL MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)
    
    # 1. Load Data Splits
    print("Loading clean dataset splits...")
    df_train = pd.read_parquet(os.path.join(DATA_SPLITS_DIR, "train_sample.parquet"))
    df_val = pd.read_parquet(os.path.join(DATA_SPLITS_DIR, "val_sample.parquet"))
    df_test = pd.read_parquet(os.path.join(DATA_SPLITS_DIR, "test_sample.parquet"))
    
    zone_lookup = load_zone_lookup("Urban_Flow_Analytics_Zone_Dataset.csv")
    print(f"Loaded {len(df_train):,} train, {len(df_val):,} val, {len(df_test):,} test trips.")

    # 2. Feature Engineering (Pre-trip features only - zero target leakage)
    print("\nExtracting pre-trip features...")
    X_train = engineer_features(df_train, zone_lookup=zone_lookup, is_training=True)
    X_val = engineer_features(df_val, zone_lookup=zone_lookup, is_training=False)
    X_test = engineer_features(df_test, zone_lookup=zone_lookup, is_training=False)
    
    print(f"Feature matrix shape: {X_train.shape} (Features: {X_train.columns.tolist()})")
    
    # Target variables
    y_fare_train = df_train["base_fare"]
    y_fare_val = df_val["base_fare"]
    y_fare_test = df_test["base_fare"]
    
    y_dur_train = df_train["duration_minutes"]
    y_dur_val = df_val["duration_minutes"]
    y_dur_test = df_test["duration_minutes"]

    all_metrics = {"fare_prediction": {}, "duration_prediction": {}, "demand_forecasting": {}}

    # =========================================================================
    # TRACK 2.1: FARE AMOUNT PREDICTION (NO-SURPRISES UPFRONT PRICING)
    # =========================================================================
    print("\n" + "#" * 70)
    print("TRACK 2.1: FARE AMOUNT PREDICTION BENCHMARKING")
    print("#" * 70)
    
    # We sample a high-power subset (e.g. 150k for training trees) to balance speed and accuracy
    train_subset_idx = np.random.RandomState(42).choice(len(X_train), size=min(150000, len(X_train)), replace=False)
    X_tr_sub = X_train.iloc[train_subset_idx]
    y_fare_tr_sub = y_fare_train.iloc[train_subset_idx]
    
    fare_results = train_and_evaluate_models(X_tr_sub, y_fare_tr_sub, X_val, y_fare_val, task_name="fare")
    best_fare_model = fare_results["best_model"]
    
    # Final Test Set Evaluation
    test_fare_preds = np.maximum(best_fare_model.predict(X_test), 0.0)
    test_fare_metrics = evaluate_predictions(y_fare_test, test_fare_preds)
    print(f"\n>>> FINAL TEST SET EVALUATION (Fare Prediction - {fare_results['best_model_name']}):")
    print(f"    RMSE: ${test_fare_metrics['RMSE']:.2f} | MAE: ${test_fare_metrics['MAE']:.2f} | R²: {test_fare_metrics['R2']:.4f} | MAPE: {test_fare_metrics['MAPE']:.2f}%")
    
    save_model(best_fare_model, os.path.join(MODELS_DIR, "fare_prediction_model.pkl"))
    all_metrics["fare_prediction"]["validation_benchmarks"] = fare_results["results"]
    all_metrics["fare_prediction"]["best_model"] = fare_results["best_model_name"]
    all_metrics["fare_prediction"]["test_metrics"] = test_fare_metrics

    # =========================================================================
    # TRACK 2.2: ON-TIME ARRIVAL ESTIMATOR (TRIP DURATION PREDICTION)
    # =========================================================================
    print("\n" + "#" * 70)
    print("TRACK 2.2: ON-TIME ARRIVAL ESTIMATOR BENCHMARKING")
    print("#" * 70)
    
    y_dur_tr_sub = y_dur_train.iloc[train_subset_idx]
    dur_results = train_and_evaluate_models(X_tr_sub, y_dur_tr_sub, X_val, y_dur_val, task_name="duration")
    best_dur_model = dur_results["best_model"]
    
    # Final Test Set Evaluation
    test_dur_preds = np.maximum(best_dur_model.predict(X_test), 0.0)
    test_dur_metrics = evaluate_predictions(y_dur_test, test_dur_preds)
    print(f"\n>>> FINAL TEST SET EVALUATION (Duration Prediction - {dur_results['best_model_name']}):")
    print(f"    RMSE: {test_dur_metrics['RMSE']:.2f} min | MAE: {test_dur_metrics['MAE']:.2f} min | R²: {test_dur_metrics['R2']:.4f} | MAPE: {test_dur_metrics['MAPE']:.2f}%")
    
    save_model(best_dur_model, os.path.join(MODELS_DIR, "duration_prediction_model.pkl"))
    all_metrics["duration_prediction"]["validation_benchmarks"] = dur_results["results"]
    all_metrics["duration_prediction"]["best_model"] = dur_results["best_model_name"]
    all_metrics["duration_prediction"]["test_metrics"] = test_dur_metrics

    # =========================================================================
    # TRACK 3.1: FLEET DISPATCHER DEMAND FORECASTING (24H / 48H / 72H)
    # =========================================================================
    print("\n" + "#" * 70)
    print("TRACK 3.1: FLEET DISPATCHER DEMAND FORECASTING")
    print("#" * 70)
    
    print("Building zone hourly time series...")
    ts_train = build_hourly_demand_series(df_train, top_n_zones=len(TOP_ZONES))
    ts_val = build_hourly_demand_series(df_val, top_n_zones=len(TOP_ZONES))
    
    ts_train_feat = create_time_series_features(ts_train)
    ts_val_feat = create_time_series_features(ts_val)
    
    dispatch_results = train_demand_forecast_model(ts_train_feat, ts_val_feat, horizons=[24, 48, 72])
    save_model(dispatch_results["model"], os.path.join(MODELS_DIR, "demand_forecaster_model.pkl"))
    all_metrics["demand_forecasting"] = dispatch_results["eval_results"]

    # Save all metrics
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll benchmark metrics saved to {METRICS_JSON}")

    # Generate Feature Importance Summary for Report
    try:
        importances = best_fare_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        feature_importance_df.to_csv(os.path.join(MODELS_DIR, "fare_feature_importances.csv"), index=False)
        print("\nTop 5 Drivers of Upfront Base Fare:")
        print(feature_importance_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"Note on feature importances: {e}")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE: ALL MODELS & METRICS READY!")
    print("=" * 70)

if __name__ == "__main__":
    run_training_pipeline()
