"""
Fleet Dispatcher Demand Forecasting Module (Track 3.1)
Builds hourly zone-level time-series models for top demand taxi zones to forecast
pickup volumes 24, 48, and 72 hours ahead, minimizing empty cruising and rider wait times.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

TOP_ZONES = [161, 237, 236, 230, 132, 138] # Midtown Center, Upper East Side, Times Sq, JFK, LGA

def build_hourly_demand_series(df: pd.DataFrame, top_n_zones: int = 5) -> pd.DataFrame:
    """
    Aggregates raw trips into an hourly pickup demand time-series per zone.
    """
    df_copy = df.copy()
    df_copy["pickup_timestamp"] = pd.to_datetime(df_copy["pickup_timestamp"], errors="coerce")
    df_copy = df_copy[df_copy["pickup_timestamp"].notna()]
    
    # Floor to hour
    df_copy["pickup_hour_dt"] = df_copy["pickup_timestamp"].dt.floor("h")
    
    # Filter to top zones
    top_zone_ids = df_copy["origin_loc_id"].value_counts().head(top_n_zones).index.tolist()
    df_top = df_copy[df_copy["origin_loc_id"].isin(top_zone_ids)]
    
    # Group by hour and zone
    demand_series = df_top.groupby(["pickup_hour_dt", "origin_loc_id"]).size().reset_index(name="demand")
    
    # Pivot so each zone is a column, or full long format with continuous hourly index
    all_hours = pd.date_range(
        start=demand_series["pickup_hour_dt"].min(),
        end=demand_series["pickup_hour_dt"].max(),
        freq="h"
    )
    
    full_dfs = []
    for zid in top_zone_ids:
        z_df = demand_series[demand_series["origin_loc_id"] == zid].set_index("pickup_hour_dt")
        z_df = z_df.reindex(all_hours, fill_value=0)
        z_df["origin_loc_id"] = zid
        z_df["pickup_hour_dt"] = all_hours
        full_dfs.append(z_df)
        
    return pd.concat(full_dfs, ignore_index=True)

def create_time_series_features(df_series: pd.DataFrame) -> pd.DataFrame:
    """
    Generates autoregressive lag features, rolling statistics, and cyclical calendar attributes.
    """
    df = df_series.sort_values(["origin_loc_id", "pickup_hour_dt"]).copy()
    
    # Calendar features
    dt = df["pickup_hour_dt"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_day"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_day"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    
    # Grouped lag features per zone
    grouped = df.groupby("origin_loc_id")["demand"]
    
    # Lags: 1h, 2h, 3h, 24h (yesterday), 48h (2 days ago), 168h (1 week ago)
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"lag_{lag}"] = grouped.shift(lag)
        
    # Rolling statistics over 24h
    df["rolling_mean_24h"] = grouped.shift(1).rolling(24, min_periods=1).mean()
    df["rolling_std_24h"] = grouped.shift(1).rolling(24, min_periods=1).std().fillna(0)
    df["rolling_max_24h"] = grouped.shift(1).rolling(24, min_periods=1).max()
    
    # Rolling mean over 7 days (168h)
    df["rolling_mean_7d"] = grouped.shift(1).rolling(168, min_periods=1).mean()
    
    return df.dropna().reset_index(drop=True)

def train_demand_forecast_model(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    horizons: list = [24, 48, 72]
) -> dict:
    """
    Trains LightGBM multi-step time series demand forecasters for 24h, 48h, and 72h horizons.
    """
    feature_cols = [
        "origin_loc_id", "hour", "dayofweek", "is_weekend",
        "sin_hour", "cos_hour", "sin_day", "cos_day",
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
        "rolling_mean_24h", "rolling_std_24h", "rolling_max_24h", "rolling_mean_7d"
    ]
    
    X_train = train_features[feature_cols]
    y_train = train_features["demand"]
    
    X_val = val_features[feature_cols]
    y_val = val_features["demand"]
    
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting Fleet Dispatcher Demand Forecaster...")
    model.fit(X_train, y_train)
    
    # Evaluate across forecast horizons
    eval_results = {}
    preds = np.maximum(model.predict(X_val), 0.0)
    
    for h in horizons:
        # Evaluate performance on slice up to horizon h
        val_slice = y_val.iloc[:h * len(TOP_ZONES)]
        preds_slice = preds[:h * len(TOP_ZONES)]
        
        rmse = np.sqrt(mean_squared_error(val_slice, preds_slice))
        mae = mean_absolute_error(val_slice, preds_slice)
        wape = (np.sum(np.abs(val_slice - preds_slice)) / max(np.sum(val_slice), 1)) * 100
        
        eval_results[f"{h}h"] = {
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "WAPE_pct": round(float(wape), 2)
        }
        print(f"Horizon {h}h Demand Forecast -> RMSE: {rmse:.2f} | MAE: {mae:.2f} trips/hr | WAPE: {wape:.2f}%")
        
    return {
        "model": model,
        "feature_cols": feature_cols,
        "eval_results": eval_results
    }
