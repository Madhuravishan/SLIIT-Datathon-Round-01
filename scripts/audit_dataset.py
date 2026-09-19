"""
Comprehensive Anomaly Audit Script for Urban Flow Analytics Dataset
Processes all 48,601,782 records across all 12 monthly CSV files in chunks
to quantify exact counts and percentages of operational anomalies.
Outputs a structured summary JSON and Markdown table for reports and notebooks.
"""

import os
import glob
import json
import time
import pandas as pd
import numpy as np

DATA_DIR = os.path.join("Urban_Flow_Analytics_Dataset_csv", "Urban_Flow_Analytics_Dataset_csv")
OUTPUT_JSON = "data_cleaning_audit.json"
OUTPUT_MD = "data_cleaning_audit.md"

def audit_full_dataset(chunksize=250000, max_files=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if max_files:
        files = files[:max_files]
        
    print(f"Found {len(files)} CSV files to audit.")
    start_time = time.time()
    
    total_records = 0
    anomalies = {
        "negative_base_fare": 0,
        "negative_charge_total": 0,
        "zero_distance_positive_fare": 0,
        "zero_or_null_rider_count": 0,
        "dropoff_before_pickup": 0,
        "unrealistic_speed_over_80mph": 0,
        "unknown_or_outside_origin_zone": 0,
        "unknown_or_outside_dest_zone": 0,
        "extreme_fare_over_500": 0,
        "extreme_distance_over_100mi": 0,
        "extreme_duration_over_24h": 0,
    }
    
    file_summaries = []

    for file_idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        print(f"[{file_idx}/{len(files)}] Auditing {filename}...", flush=True)
        file_records = 0
        file_anomalies = {k: 0 for k in anomalies}
        
        # Read in chunks
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=chunksize,
            usecols=[
                "pickup_timestamp", "dropoff_timestamp", "rider_count",
                "distance_miles", "origin_loc_id", "dest_loc_id",
                "base_fare", "charge_total"
            ]
        )
        
        for chunk in chunk_iter:
            n_chunk = len(chunk)
            file_records += n_chunk
            
            # 1. Negative fares
            neg_bf = (chunk["base_fare"] < 0).sum()
            neg_ct = (chunk["charge_total"] < 0).sum()
            file_anomalies["negative_base_fare"] += int(neg_bf)
            file_anomalies["negative_charge_total"] += int(neg_ct)
            
            # 2. Distance == 0 & base_fare > 0
            zero_dist_pos_fare = ((chunk["distance_miles"] == 0) & (chunk["base_fare"] > 0)).sum()
            file_anomalies["zero_distance_positive_fare"] += int(zero_dist_pos_fare)
            
            # 3. Rider count == 0 or null
            zero_null_riders = ((chunk["rider_count"] == 0) | (chunk["rider_count"].isna())).sum()
            file_anomalies["zero_or_null_rider_count"] += int(zero_null_riders)
            
            # 4 & 5. Fast temporal duration
            pickup_dt = pd.to_datetime(chunk["pickup_timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            dropoff_dt = pd.to_datetime(chunk["dropoff_timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            duration_sec = (dropoff_dt - pickup_dt).dt.total_seconds().fillna(0)
            
            dropoff_before = (duration_sec <= 0).sum()
            file_anomalies["dropoff_before_pickup"] += int(dropoff_before)
            
            # Speeds (distance in miles / hours)
            valid_duration = duration_sec > 0
            hours = duration_sec / 3600.0
            # calculate speed only for positive duration
            speed = np.where(valid_duration, chunk["distance_miles"] / hours, 0.0)
            speed_over_80 = (speed > 80.0).sum()
            file_anomalies["unrealistic_speed_over_80mph"] += int(speed_over_80)
            
            # 6. Unknown / outside zones (loc_id 264=Unknown, 265=Outside of NYC)
            unk_origin = (chunk["origin_loc_id"].isin([264, 265])).sum()
            unk_dest = (chunk["dest_loc_id"].isin([264, 265])).sum()
            file_anomalies["unknown_or_outside_origin_zone"] += int(unk_origin)
            file_anomalies["unknown_or_outside_dest_zone"] += int(unk_dest)
            
            # Extreme outliers
            file_anomalies["extreme_fare_over_500"] += int((chunk["base_fare"] > 500).sum())
            file_anomalies["extreme_distance_over_100mi"] += int((chunk["distance_miles"] > 100).sum())
            file_anomalies["extreme_duration_over_24h"] += int((duration_sec > 86400).sum())
            
        total_records += file_records
        for k in anomalies:
            anomalies[k] += file_anomalies[k]
            
        file_summaries.append({
            "file": filename,
            "records": file_records,
            "anomalies": file_anomalies
        })
        print(f"  Processed {file_records:,} rows (Cumulative: {total_records:,})", flush=True)

    elapsed = time.time() - start_time
    print(f"\nAudit completed in {elapsed:.1f} seconds! Total records: {total_records:,}")

    # Compile report
    results = {
        "total_records": total_records,
        "elapsed_seconds": round(elapsed, 2),
        "anomalies": {}
    }

    policies = {
        "negative_base_fare": ("Filter / Drop", "Disputed, voided, or refunded transactions; cannot represent valid ride fare."),
        "negative_charge_total": ("Filter / Drop", "Total charge negative due to reversals/chargebacks."),
        "zero_distance_positive_fare": ("Filter / Segment", "Trips where vehicle did not move (cancellation/waiting fees, GPS initialization failure). Not representative of point-to-point transit pricing."),
        "zero_or_null_rider_count": ("Impute (Mode = 1)", "Unmetered passenger entries, courier delivery, or driver input oversight. Valid trips with realistic distance and fare should be retained with imputed passenger count."),
        "dropoff_before_pickup": ("Filter / Drop", "Clock synchronization error or instant trip cancellation resulting in negative or zero duration."),
        "unrealistic_speed_over_80mph": ("Filter / Drop", "GPS teleportation or inaccurate timestamp entry violating physical urban driving limits."),
        "unknown_or_outside_origin_zone": ("Segment / Keep separate", "Trips originating outside NYC jurisdiction or with unmapped zone IDs; exclude from intra-city spatial flow models."),
        "unknown_or_outside_dest_zone": ("Segment / Keep separate", "Trips ending outside NYC jurisdiction or unmapped zone IDs."),
        "extreme_fare_over_500": ("Filter / Drop", "Outlier fares exceeding standard maximum thresholds; severe skewness risk for regression."),
        "extreme_distance_over_100mi": ("Filter / Drop", "Long-distance inter-city trips outside standard metropolitan fleet operational envelope."),
        "extreme_duration_over_24h": ("Filter / Drop", "Meter left running overnight or multi-day logging error."),
    }

    print("\n" + "=" * 85)
    print(f"{'ANOMALY CATEGORY':<35} | {'COUNT':<10} | {'PERCENT':<8} | {'POLICY':<15}")
    print("=" * 85)

    md_table = [
        "| Anomaly Category | Count | Percentage | Justification & Policy |",
        "| :--- | :---: | :---: | :--- |"
    ]

    for k, v in anomalies.items():
        pct = (v / total_records * 100) if total_records > 0 else 0
        policy, reason = policies.get(k, ("Filter", "Operational anomaly"))
        results["anomalies"][k] = {
            "count": v,
            "percentage": round(pct, 4),
            "policy": policy,
            "justification": reason
        }
        print(f"{k:<35} | {v:<10,d} | {pct:>6.3f}% | {policy:<15}")
        md_table.append(f"| **{k.replace('_', ' ').title()}** | {v:,} | {pct:.3f}% | **{policy}**: {reason} |")

    print("=" * 85)

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

    # Save Markdown table
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Urban Flow Analytics - Full Dataset Anomaly Audit\n\n")
        f.write(f"**Total Records Audited:** {total_records:,} trips across 12 monthly files (April 2025 – March 2026)\n\n")
        f.write("\n".join(md_table))
        f.write("\n")
    print(f"Markdown report saved to {OUTPUT_MD}")

    return results

if __name__ == "__main__":
    audit_full_dataset()
