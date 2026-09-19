"""
Hotspot & Origin-Destination Flow Clustering Module (Track 3.2)
Analyzes macro travel corridors across 4 core daily time slices:
Morning Commute (7-10), Midday (11-15), Evening Rush (17-20), and Late Night (22-04).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

TIME_SLICES = {
    "Morning Commute (07:00 - 10:00)": (7, 10),
    "Midday Commercial (11:00 - 15:00)": (11, 15),
    "Evening Rush (17:00 - 20:00)": (17, 20),
    "Late Night (22:00 - 04:00)": (22, 4)
}

def assign_time_slice(hour: int) -> str:
    """Classifies an hour into one of the 4 standard operational time slices."""
    if 7 <= hour <= 10:
        return "Morning Commute (07:00 - 10:00)"
    elif 11 <= hour <= 15:
        return "Midday Commercial (11:00 - 15:00)"
    elif 17 <= hour <= 20:
        return "Evening Rush (17:00 - 20:00)"
    elif hour >= 22 or hour <= 4:
        return "Late Night (22:00 - 04:00)"
    else:
        return "Transitional / Off-Peak"

def analyze_od_flows(
    df: pd.DataFrame,
    zone_lookup: dict = None,
    top_n_corridors: int = 10
) -> dict:
    """
    Computes Origin-Destination flow volumes, average fares, and travel times
    stratified by the 4 operational daily time slices.
    """
    df_copy = df.copy()
    pickup_dt = pd.to_datetime(df_copy["pickup_timestamp"], errors="coerce")
    df_copy = df_copy[pickup_dt.notna()].copy()
    df_copy["hour"] = pickup_dt[pickup_dt.notna()].dt.hour
    df_copy["time_slice"] = df_copy["hour"].apply(assign_time_slice)
    
    slice_results = {}
    
    for slice_name in TIME_SLICES.keys():
        slice_df = df_copy[df_copy["time_slice"] == slice_name]
        
        # OD corridor grouping
        corridors = slice_df.groupby(["origin_loc_id", "dest_loc_id"]).agg(
            trip_volume=("base_fare", "count"),
            avg_base_fare=("base_fare", "mean"),
            avg_distance_miles=("distance_miles", "mean"),
            avg_duration_min=("duration_minutes", "mean") if "duration_minutes" in slice_df.columns else ("base_fare", lambda x: 0)
        ).reset_index()
        
        corridors = corridors.sort_values("trip_volume", ascending=False).head(top_n_corridors)
        
        # Enrich with zone names if lookup available
        if zone_lookup:
            corridors["origin_zone"] = corridors["origin_loc_id"].map(lambda z: zone_lookup.get(z, {}).get("zone_name", f"Zone {z}"))
            corridors["dest_zone"] = corridors["dest_loc_id"].map(lambda z: zone_lookup.get(z, {}).get("zone_name", f"Zone {z}"))
            corridors["origin_borough"] = corridors["origin_loc_id"].map(lambda z: zone_lookup.get(z, {}).get("borough_name", "Unknown"))
            corridors["dest_borough"] = corridors["dest_loc_id"].map(lambda z: zone_lookup.get(z, {}).get("borough_name", "Unknown"))
            corridors["corridor_name"] = corridors["origin_zone"] + " -> " + corridors["dest_zone"]
        else:
            corridors["corridor_name"] = corridors["origin_loc_id"].astype(str) + " -> " + corridors["dest_loc_id"].astype(str)
            
        slice_results[slice_name] = corridors
        
    return slice_results

def cluster_demand_hotspots(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Performs spatial demand clustering across zones based on volume, fare, and distance profiles.
    """
    zone_stats = df.groupby("origin_loc_id").agg(
        total_trips=("base_fare", "count"),
        avg_fare=("base_fare", "mean"),
        avg_distance=("distance_miles", "mean")
    ).reset_index()
    
    features = zone_stats[["total_trips", "avg_fare", "avg_distance"]]
    # Normalize features
    norm_features = (features - features.mean()) / (features.std() + 1e-8)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    zone_stats["cluster_id"] = kmeans.fit_predict(norm_features)
    
    cluster_labels = {
        0: "High Volume / Short Midtown Trips",
        1: "Airport Corridor / Long Haul",
        2: "Low Density / Residential Outer Borough",
        3: "Moderate Volume / Cross-Borough Commute",
        4: "Downtown Financial / Core Commercial"
    }
    zone_stats["cluster_label"] = zone_stats["cluster_id"].map(lambda c: cluster_labels.get(c, f"Cluster {c}"))
    
    return zone_stats
