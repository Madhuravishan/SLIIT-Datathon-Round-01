"""
Data Cleaning & Preprocessing Pipeline for Urban Flow Analytics
Enforces rigorous filtering policies for anomalies, handles imputation,
and constructs leak-free temporal Train/Val/Test splits.
"""

import os
import glob
import pandas as pd
import numpy as np

VALID_ZONES = set(range(1, 264)) # Loc IDs 1 through 263 are valid NYC zones

def clean_dataframe(df: pd.DataFrame, impute_riders: bool = True) -> pd.DataFrame:
    """
    Cleans a dataframe of raw taxi records according to justified anomaly policies.
    """
    # 1. Parse timestamps and calculate duration
    pickup_dt = pd.to_datetime(df["pickup_timestamp"], errors="coerce")
    dropoff_dt = pd.to_datetime(df["dropoff_timestamp"], errors="coerce")
    
    # Filter unparseable timestamps
    valid_time = pickup_dt.notna() & dropoff_dt.notna()
    df = df[valid_time].copy()
    pickup_dt = pickup_dt[valid_time]
    dropoff_dt = dropoff_dt[valid_time]
    
    duration_sec = (dropoff_dt - pickup_dt).dt.total_seconds()
    duration_min = duration_sec / 60.0
    
    # 2. Duration filter: between 1 minute (60s) and 4 hours (14,400s)
    valid_duration = (duration_sec >= 60) & (duration_sec <= 14400)
    
    # 3. Distance filter: between 0.1 miles and 75 miles
    valid_distance = (df["distance_miles"] >= 0.1) & (df["distance_miles"] <= 75.0)
    
    # 4. Fare filter: positive base fare and charge total, maximum $300
    valid_fare = (df["base_fare"] >= 2.50) & (df["base_fare"] <= 300.0) & (df["charge_total"] > 0)
    
    # 5. Speed filter: distance / hours <= 75 mph (NYC urban context)
    hours = duration_sec / 3600.0
    speed_mph = df["distance_miles"] / hours
    valid_speed = speed_mph <= 75.0
    
    # 6. Valid zone filter: origin and destination in valid NYC zones
    valid_zones = df["origin_loc_id"].isin(VALID_ZONES) & df["dest_loc_id"].isin(VALID_ZONES)
    
    # Combine boolean masks
    clean_mask = valid_duration & valid_distance & valid_fare & valid_speed & valid_zones
    df_clean = df[clean_mask].copy()
    
    # Add computed target duration
    df_clean["duration_minutes"] = duration_min[clean_mask].round(2)
    df_clean["speed_mph"] = speed_mph[clean_mask].round(2)
    
    # 7. Imputation: passenger count
    if impute_riders:
        df_clean["rider_count"] = df_clean["rider_count"].fillna(1.0)
        df_clean.loc[df_clean["rider_count"] <= 0, "rider_count"] = 1.0
        df_clean.loc[df_clean["rider_count"] > 6, "rider_count"] = 6.0
        
    return df_clean

def create_temporal_splits(
    data_dir: str,
    output_dir: str = "data_splits",
    sample_per_month: int = 50000,
    random_seed: int = 42
):
    """
    Constructs clean, leak-free Train (April-Dec 2025), Val (Jan-Feb 2026),
    and Test (March 2026) splits from all 12 monthly CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    train_chunks = []
    val_chunks = []
    test_chunks = []
    
    print(f"Generating stratified temporal splits from {len(files)} files...")
    
    for f in files:
        fname = os.path.basename(f)
        print(f"Reading sample from {fname}...")
        
        # Read a representative chunk from the month
        df_month = pd.read_csv(f, nrows=sample_per_month * 3)
        df_clean = clean_dataframe(df_month)
        
        # Take target sample size
        if len(df_clean) > sample_per_month:
            df_sampled = df_clean.sample(n=sample_per_month, random_state=random_seed)
        else:
            df_sampled = df_clean
            
        # Temporal partitioning:
        # Months 2025-04 to 2025-12 -> Train (9 months)
        # Months 2026-01 to 2026-02 -> Val (2 months)
        # Month 2026-03 -> Test (Final out-of-time evaluation month)
        if "2025-" in fname:
            train_chunks.append(df_sampled)
        elif "2026-01" in fname or "2026-02" in fname:
            val_chunks.append(df_sampled)
        elif "2026-03" in fname:
            test_chunks.append(df_sampled)
            
    df_train = pd.concat(train_chunks, ignore_index=True)
    df_val = pd.concat(val_chunks, ignore_index=True)
    df_test = pd.concat(test_chunks, ignore_index=True)
    
    print(f"\nSplit Sizes:")
    print(f"Train Set: {len(df_train):,} records (April 2025 - December 2025)")
    print(f"Val Set:   {len(df_val):,} records (January 2026 - February 2026)")
    print(f"Test Set:  {len(df_test):,} records (March 2026 - Out-of-time Holdout)")
    
    # Save as Parquet and CSV
    train_parquet = os.path.join(output_dir, "train_sample.parquet")
    val_parquet = os.path.join(output_dir, "val_sample.parquet")
    test_parquet = os.path.join(output_dir, "test_sample.parquet")
    
    df_train.to_parquet(train_parquet, index=False)
    df_val.to_parquet(val_parquet, index=False)
    df_test.to_parquet(test_parquet, index=False)
    
    # Also export compressed CSV for universal accessibility
    df_train.head(10000).to_csv(os.path.join(output_dir, "train_sample_preview.csv"), index=False)
    df_val.head(5000).to_csv(os.path.join(output_dir, "val_sample_preview.csv"), index=False)
    df_test.head(5000).to_csv(os.path.join(output_dir, "test_sample_preview.csv"), index=False)
    
    print(f"Splits saved to {output_dir}/ successfully!")
    return df_train, df_val, df_test

if __name__ == "__main__":
    DATA_PATH = os.path.join("Urban_Flow_Analytics_Dataset_csv", "Urban_Flow_Analytics_Dataset_csv")
    create_temporal_splits(DATA_PATH)
