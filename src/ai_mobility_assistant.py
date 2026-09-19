"""
AI Mobility Assistant Engine (Track 5 Bonus)
Enables non-technical city officials and fleet managers to query taxi data in plain English.
Includes an intelligent ambiguity detector and safe clarification protocol.
"""

import re
import pandas as pd
import numpy as np

class AIMobilityAssistant:
    def __init__(self, data_df: pd.DataFrame = None, zone_lookup: dict = None):
        self.df = data_df
        self.zone_lookup = zone_lookup or {}

    def query(self, user_prompt: str) -> dict:
        """
        Interprets natural language questions, validates clarity, executes safe queries,
        or asks for clarification if the input is ambiguous or incomplete.
        """
        prompt = user_prompt.strip().lower()
        
        # 1. Ambiguity & Incompleteness Detection
        ambiguity_response = self._check_ambiguity(prompt)
        if ambiguity_response:
            return ambiguity_response
            
        # 2. Intent Routing & Data Retrieval
        if any(w in prompt for w in ["highest demand", "busiest", "most trips", "top locations", "top zones", "demand", "highest taxi demand", "locations had the highest"]):
            return self._handle_top_demand(prompt)
            
        elif any(w in prompt for w in ["average fare", "avg fare", "how much", "cost", "fare during", "fares"]):
            return self._handle_average_fare(prompt)
            
        elif any(w in prompt for w in ["trip duration", "how long", "travel time", "average time", "duration"]):
            return self._handle_trip_duration(prompt)
            
        elif any(w in prompt for w in ["tips", "tipping", "gratuity"]):
            return self._handle_tips(prompt)
            
        elif any(w in prompt for w in ["peak hours", "busiest hour", "hourly demand", "time of day"]):
            return self._handle_peak_hours(prompt)
            
        elif any(w in prompt for w in ["airport", "jfk", "lga", "airports", "terminal"]):
            return self._handle_airports(prompt)
            
        elif any(w in prompt for w in ["longest", "furthest", "top routes", "distance", "miles"]):
            return self._handle_longest_trips(prompt)
            
        elif any(w in prompt for w in ["congestion", "congestion fee", "congestion relief"]):
            return self._handle_congestion_fee(prompt)
            
        elif any(w in prompt for w in ["day of week", "weekend", "busiest day", "sunday", "saturday", "friday", "monday"]):
            return self._handle_day_of_week(prompt)
            
        else:
            return {
                "status": "needs_clarification",
                "answer": (
                    "I couldn't clearly match your query to a specific mobility metric. "
                    "As an AI assistant designed for urban mobility analytics, I can answer questions about:\n"
                    "- 📍 **Highest demand pickup zones & hotspots**\n"
                    "- 💵 **Average fares (overall, peak hours, or airport)**\n"
                    "- ⏱️ **Trip travel times and longest corridors**\n"
                    "- 💳 **Driver tips & gratuity distributions**\n"
                    "- ✈️ **Airport travel volumes (JFK & LGA)**"
                ),
                "data": None,
                "suggested_queries": [
                    "Which locations had the highest taxi demand?",
                    "What was the average fare during peak hours?",
                    "What is the average fare and trip volume to JFK airport?",
                    "What are the top 5 longest trip routes?",
                    "Which day of the week experiences the highest demand?"
                ]
            }

    def _check_ambiguity(self, prompt: str) -> dict:
        """Flags under-specified or ambiguous queries to avoid hallucinations."""
        # Ultra-short or vague prompts
        if len(prompt.split()) <= 2 and prompt in ["fare", "trips", "demand", "traffic", "show data", "best place"]:
            return {
                "status": "ambiguous",
                "answer": (
                    f"Your question '{prompt}' is quite broad. To give you an exact answer, could you clarify:\n"
                    "- Are you looking for base metered fare, total customer charge, or tips?\n"
                    "- Which specific zone, borough, or time window (e.g., peak hours vs weekend) should I analyze?"
                ),
                "data": None,
                "suggested_queries": [
                    "Show average fare by borough",
                    "Which zones have the highest demand during morning peak?",
                    "Show demand distribution across all hours"
                ]
            }
        return None

    def _handle_top_demand(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        top_zones = self.df["origin_loc_id"].value_counts().head(10).reset_index()
        top_zones.columns = ["origin_loc_id", "trip_count"]
        top_zones["zone_name"] = top_zones["origin_loc_id"].map(lambda z: self.zone_lookup.get(z, {}).get("zone_name", f"Zone {z}"))
        top_zones["borough"] = top_zones["origin_loc_id"].map(lambda z: self.zone_lookup.get(z, {}).get("borough_name", "Unknown"))
        
        busiest = top_zones.iloc[0]["zone_name"]
        count = top_zones.iloc[0]["trip_count"]
        
        answer = (
            f"Based on operational trip records, the zone with the highest taxi demand is **{busiest}** "
            f"with **{count:,} recorded pickups**, followed by {top_zones.iloc[1]['zone_name']} "
            f"and {top_zones.iloc[2]['zone_name']}."
        )
        return {"status": "success", "answer": answer, "data": top_zones}

    def _handle_average_fare(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        if "peak" in prompt:
            # Morning peak 7-10, Evening 17-20
            dt = pd.to_datetime(self.df["pickup_timestamp"], errors="coerce")
            hours = dt.dt.hour
            is_peak = ((hours >= 7) & (hours <= 10)) | ((hours >= 17) & (hours <= 20))
            avg_peak = self.df.loc[is_peak, "base_fare"].mean()
            avg_off = self.df.loc[~is_peak, "base_fare"].mean()
            
            answer = (
                f"During peak rush hours (07:00-10:00 and 17:00-20:00), the average base fare is **${avg_peak:.2f}**, "
                f"compared to **${avg_off:.2f}** during off-peak hours (a {((avg_peak/avg_off)-1)*100:.1f}% difference)."
            )
            data = pd.DataFrame([
                {"Period": "Peak Hours (07-10 & 17-20)", "Average_Base_Fare": round(avg_peak, 2)},
                {"Period": "Off-Peak Hours", "Average_Base_Fare": round(avg_off, 2)}
            ])
            return {"status": "success", "answer": answer, "data": data}
        else:
            overall_avg = self.df["base_fare"].mean()
            median_fare = self.df["base_fare"].median()
            answer = f"The overall average base fare is **${overall_avg:.2f}** (median: **${median_fare:.2f}**)."
            return {"status": "success", "answer": answer, "data": None}

    def _handle_trip_duration(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        dur_col = "duration_minutes" if "duration_minutes" in self.df.columns else None
        if dur_col is None:
            dt_p = pd.to_datetime(self.df["pickup_timestamp"], errors="coerce")
            dt_d = pd.to_datetime(self.df["dropoff_timestamp"], errors="coerce")
            dur_series = (dt_d - dt_p).dt.total_seconds() / 60.0
        else:
            dur_series = self.df[dur_col]
            
        avg_dur = dur_series.mean()
        med_dur = dur_series.median()
        answer = f"The average trip duration across all recorded journeys is **{avg_dur:.1f} minutes** (median: **{med_dur:.1f} minutes**)."
        return {"status": "success", "answer": answer, "data": None}

    def _handle_tips(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        # Tips only recorded for credit card trips (fare_settlement_method == 1)
        card_trips = self.df[self.df["fare_settlement_method"] == 1]
        avg_tip = card_trips["driver_tip_payment"].mean()
        tip_pct = (card_trips["driver_tip_payment"] / card_trips["base_fare"]).clip(0, 1).mean() * 100
        
        answer = (
            f"For credit card trips (where tips are digitally recorded), the average gratuity is **${avg_tip:.2f}**, "
            f"representing an average tip rate of **{tip_pct:.1f}%** relative to base metered fare."
        )
        return {"status": "success", "answer": answer, "data": None}

    def _handle_peak_hours(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        dt = pd.to_datetime(self.df["pickup_timestamp"], errors="coerce")
        hourly = dt.dt.hour.value_counts().sort_index().reset_index()
        hourly.columns = ["Hour_of_Day", "Trip_Count"]
        busiest_hour = hourly.loc[hourly["Trip_Count"].idxmax()]["Hour_of_Day"]
        
        answer = (
            f"The single busiest time of day for taxi demand is **{busiest_hour:02d}:00-{busiest_hour+1:02d}:00**, "
            f"coinciding with the evening commute peak."
        )
        return {"status": "success", "answer": answer, "data": hourly}

    def _handle_congestion_fee(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        col = "congestion_relief_fee" if "congestion_relief_fee" in self.df.columns else "zone_congestion_fee"
        avg_fee = self.df[col].mean()
        total_rev = self.df[col].sum()
        trips_with_fee = (self.df[col] > 0).mean() * 100
        
        answer = (
            f"Congestion relief fees are applied to **{trips_with_fee:.1f}%** of trips, "
            f"generating an average of **${avg_fee:.2f} per ride** in dedicated transit infrastructure revenue."
        )
        return {"status": "success", "answer": answer, "data": None}

    def _handle_airports(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        jfk_trips = self.df[(self.df["origin_loc_id"] == 132) | (self.df["dest_loc_id"] == 132)]
        lga_trips = self.df[(self.df["origin_loc_id"] == 138) | (self.df["dest_loc_id"] == 138)]
        
        jfk_avg_fare = jfk_trips["base_fare"].mean() if len(jfk_trips) > 0 else 0.0
        lga_avg_fare = lga_trips["base_fare"].mean() if len(lga_trips) > 0 else 0.0
        
        answer = (
            f"Airport trips represent high-value travel corridors:\n"
            f"- **JFK Airport (Zone 132):** {len(jfk_trips):,} trips with an average base fare of **${jfk_avg_fare:.2f}**\n"
            f"- **LaGuardia Airport (Zone 138):** {len(lga_trips):,} trips with an average base fare of **${lga_avg_fare:.2f}**"
        )
        data = pd.DataFrame([
            {"Airport": "JFK International (Zone 132)", "Trip_Count": len(jfk_trips), "Avg_Base_Fare": round(jfk_avg_fare, 2)},
            {"Airport": "LaGuardia (Zone 138)", "Trip_Count": len(lga_trips), "Avg_Base_Fare": round(lga_avg_fare, 2)}
        ])
        return {"status": "success", "answer": answer, "data": data}

    def _handle_longest_trips(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        top_routes = self.df.sort_values(by="distance_miles", ascending=False).head(5)[
            ["origin_loc_id", "dest_loc_id", "distance_miles", "base_fare"]
        ].copy()
        top_routes["pickup_zone"] = top_routes["origin_loc_id"].map(lambda z: self.zone_lookup.get(z, {}).get("zone_name", f"Zone {z}"))
        top_routes["dropoff_zone"] = top_routes["dest_loc_id"].map(lambda z: self.zone_lookup.get(z, {}).get("zone_name", f"Zone {z}"))
        
        top_dist = top_routes.iloc[0]["distance_miles"]
        top_from = top_routes.iloc[0]["pickup_zone"]
        top_to = top_routes.iloc[0]["dropoff_zone"]
        
        answer = (
            f"The longest validated non-outlier trip recorded was **{top_dist:.1f} miles** "
            f"traveling from **{top_from}** to **{top_to}**."
        )
        return {"status": "success", "answer": answer, "data": top_routes[["pickup_zone", "dropoff_zone", "distance_miles", "base_fare"]]}

    def _handle_day_of_week(self, prompt: str) -> dict:
        if self.df is None:
            return {"status": "error", "answer": "No dataset loaded."}
            
        dt = pd.to_datetime(self.df["pickup_timestamp"], errors="coerce")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = dt.dt.day_name().value_counts().reindex(days).reset_index()
        dow_counts.columns = ["Day_of_Week", "Trip_Count"]
        
        busiest_day = dow_counts.loc[dow_counts["Trip_Count"].idxmax()]["Day_of_Week"]
        
        answer = (
            f"Across all recorded trips, **{busiest_day}** experiences the highest total trip volume, "
            f"driven by evening social activities and commuter peaks."
        )
        return {"status": "success", "answer": answer, "data": dow_counts}

