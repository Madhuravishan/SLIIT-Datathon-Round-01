"""
Feature Engineering Pipeline for Upfront Base Fare and Trip Duration Prediction
Extracts spatial, temporal, cyclical, and operational features without target leakage.
"""

import os
import pandas as pd
import numpy as np

# Key NYC Airport Zone IDs
AIRPORT_ZONES = {
    1: "EWR",
    132: "JFK",
    138: "LGA"
}

def load_zone_lookup(zone_csv_path: str = "Urban_Flow_Analytics_Zone_Dataset.csv") -> dict:
    """Loads zone lookup table mapping loc_id to borough and zone details."""
    if not os.path.exists(zone_csv_path):
        return {}
    df_zone = pd.read_csv(zone_csv_path)
    return df_zone.set_index("loc_id").to_dict(orient="index")

def engineer_features(
    df: pd.DataFrame,
    zone_lookup: dict = None,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Builds clean, leakage-free feature matrix for upfront pricing and duration prediction.
    Only pre-trip features available at booking time are utilized.
    """
    X = pd.DataFrame(index=df.index)
    
    # 1. Temporal Features from Pickup Timestamp
    pickup_dt = pd.to_datetime(df["pickup_timestamp"], errors="coerce")
    
    X["pickup_hour"] = pickup_dt.dt.hour.fillna(12).astype(int)
    X["pickup_dayofweek"] = pickup_dt.dt.dayofweek.fillna(2).astype(int)
    X["pickup_month"] = pickup_dt.dt.month.fillna(6).astype(int)
    X["is_weekend"] = (X["pickup_dayofweek"] >= 5).astype(int)
    
    # Peak hour indicators (Morning: 7-10, Evening: 17-20)
    X["is_morning_peak"] = ((X["pickup_hour"] >= 7) & (X["pickup_hour"] <= 10)).astype(int)
    X["is_evening_peak"] = ((X["pickup_hour"] >= 17) & (X["pickup_hour"] <= 20)).astype(int)
    X["is_night"] = ((X["pickup_hour"] >= 22) | (X["pickup_hour"] <= 4)).astype(int)
    
    # Cyclical hour encoding
    X["sin_hour"] = np.sin(2 * np.pi * X["pickup_hour"] / 24.0)
    X["cos_hour"] = np.cos(2 * np.pi * X["pickup_hour"] / 24.0)
    X["sin_day"] = np.sin(2 * np.pi * X["pickup_dayofweek"] / 7.0)
    X["cos_day"] = np.cos(2 * np.pi * X["pickup_dayofweek"] / 7.0)
    
    # 2. Distance Features
    X["distance_miles"] = df["distance_miles"].fillna(1.0).astype(float)
    X["log_distance"] = np.log1p(np.maximum(X["distance_miles"], 0.0))
    X["sqrt_distance"] = np.sqrt(np.maximum(X["distance_miles"], 0.0))
    
    # 3. Spatial & Zone Features
    X["origin_loc_id"] = df["origin_loc_id"].fillna(0).astype(int)
    X["dest_loc_id"] = df["dest_loc_id"].fillna(0).astype(int)
    
    # Airport Corridor Flags
    X["is_airport_origin"] = X["origin_loc_id"].isin(AIRPORT_ZONES.keys()).astype(int)
    X["is_airport_dest"] = X["dest_loc_id"].isin(AIRPORT_ZONES.keys()).astype(int)
    X["is_airport_trip"] = (X["is_airport_origin"] | X["is_airport_dest"]).astype(int)
    
    # Zone enrichment if lookup provided
    if zone_lookup:
        origin_borough = df["origin_loc_id"].map(lambda x: zone_lookup.get(x, {}).get("borough_name", "Unknown"))
        dest_borough = df["dest_loc_id"].map(lambda x: zone_lookup.get(x, {}).get("borough_name", "Unknown"))
        
        X["is_same_borough"] = (origin_borough == dest_borough).astype(int)
        X["is_manhattan_core"] = ((origin_borough == "Manhattan") & (dest_borough == "Manhattan")).astype(int)
    else:
        X["is_same_borough"] = (X["origin_loc_id"] == X["dest_loc_id"]).astype(int)
        X["is_manhattan_core"] = 0
        
    # 4. Operational Attributes
    X["rider_count"] = df["rider_count"].fillna(1.0).clip(1, 6).astype(float)
    X["rate_class_id"] = df["rate_class_id"].fillna(1.0).astype(int)
    X["provider_code"] = df["provider_code"].fillna(1).astype(int)
    
    # Rate class specific indicators
    X["is_jfk_flat_rate"] = (X["rate_class_id"] == 2).astype(int)
    X["is_negotiated_fare"] = (X["rate_class_id"] == 5).astype(int)
    
    return X
