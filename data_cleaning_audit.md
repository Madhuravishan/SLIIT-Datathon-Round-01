# Urban Flow Analytics - Full Dataset Anomaly Audit

**Total Records Audited:** 48,601,782 trips across 12 monthly files (April 2025 – March 2026)

| Anomaly Category | Count | Percentage | Justification & Policy |
| :--- | :---: | :---: | :--- |
| **Negative Base Fare** | 2,400,031 | 4.938% | **Filter / Drop**: Disputed, voided, or refunded transactions; cannot represent valid ride fare. |
| **Negative Charge Total** | 875,399 | 1.801% | **Filter / Drop**: Total charge negative due to reversals/chargebacks. |
| **Zero Distance Positive Fare** | 1,267,110 | 2.607% | **Filter / Segment**: Trips where vehicle did not move (cancellation/waiting fees, GPS initialization failure). Not representative of point-to-point transit pricing. |
| **Zero Or Null Rider Count** | 12,636,846 | 26.001% | **Impute (Mode = 1)**: Unmetered passenger entries, courier delivery, or driver input oversight. Valid trips with realistic distance and fare should be retained with imputed passenger count. |
| **Dropoff Before Pickup** | 651,610 | 1.341% | **Filter / Drop**: Clock synchronization error or instant trip cancellation resulting in negative or zero duration. |
| **Unrealistic Speed Over 80Mph** | 13,589 | 0.028% | **Filter / Drop**: GPS teleportation or inaccurate timestamp entry violating physical urban driving limits. |
| **Unknown Or Outside Origin Zone** | 92,853 | 0.191% | **Segment / Keep separate**: Trips originating outside NYC jurisdiction or with unmapped zone IDs; exclude from intra-city spatial flow models. |
| **Unknown Or Outside Dest Zone** | 317,258 | 0.653% | **Segment / Keep separate**: Trips ending outside NYC jurisdiction or unmapped zone IDs. |
| **Extreme Fare Over 500** | 860 | 0.002% | **Filter / Drop**: Outlier fares exceeding standard maximum thresholds; severe skewness risk for regression. |
| **Extreme Distance Over 100Mi** | 2,817 | 0.006% | **Filter / Drop**: Long-distance inter-city trips outside standard metropolitan fleet operational envelope. |
| **Extreme Duration Over 24H** | 405 | 0.001% | **Filter / Drop**: Meter left running overnight or multi-day logging error. |
