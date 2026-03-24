# Candidate Pair Findings (Top 20)

Source scores: `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv` (filtered to plausible MMSI IDs)

- Proximity validation: `artifacts/top100_overlap_summary_daily_full_25km_w1.csv`, `artifacts/top100_overlap_summary_daily_full_50km_w3.csv`, `artifacts/top100_overlap_summary_daily_full_100km_w7.csv`
- Region validation: `artifacts/top100_overlap_summary_daily_full_region2deg.csv`
- Optional flag/gear (Aug 2017 TGCN enrichment, merged when pair matches): `artifacts/cooperative_pairs_with_flag_gear.csv`

|src|dst|score|overlap_days|within25km_±1d|mean_dist_25km|within50km_±3d|mean_dist_50km|within100km_±7d|mean_dist_100km|region2deg_overlap_ratio|region2deg_close_ratio|src_mid|dst_mid|src_gear|dst_gear|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|257317000|316003227|15.983|1168|0|5304.12|0|5306.08|0|5308.71|0.831|0.000|||||
|813021762|412413289|15.932|1016|0|738.99|0|739.12|0|738.83|0.398|0.000|||||
|412000690|412325200|15.919|774|102|135.15|410|136.34|1568|139.02|0.579|0.230|||||
|251146940|224072680|15.904|705|0|3339.62|0|3339.49|0|3340.26|0.613|0.000|||||
|412061791|412508302|15.809|226|8|215.99|88|231.88|378|251.85|0.401|0.132|||||
|224017870|412428302|15.799|523|0|9882.37|0|9882.41|0|9882.83|0.456|0.000|||||
|412150903|237554000|15.759|141|0|8209.60|0|8201.31|0|8204.70|0.201|0.000|||||
|271072219|412671880|15.602|460|0|9128.20|0|9136.21|0|9140.82|0.342|0.000|||||
|273810110|800050093|15.550|265|0|4364.64|0|4371.21|0|4351.44|0.848|0.000|||||
|271072320|251422540|15.526|1283|0|4268.24|0|4267.96|0|4266.12|0.493|0.000|||||
|412436875|412836422|15.498|68|0|742.35|0|736.70|0|733.87|0.140|0.000|||||
|412327536|316003779|15.487|136|0|8631.99|0|8630.43|0|8623.01|0.274|0.000|||||
|205243000|412470145|15.419|1297|0|9790.84|0|9789.97|0|9786.65|0.731|0.000|||||
|224060150|412333102|15.410|118|0|10153.81|0|10154.64|0|10154.81|0.554|0.000|||||
|412322715|257391020|15.366|577|0|7076.42|0|7076.56|0|7077.10|0.824|0.000|||||
|412334569|412457054|15.278|464|0|1481.70|0|1482.50|0|1482.44|0.284|0.000|||||
|431007759|440122430|15.271|1370|0|992.24|0|992.86|0|993.79|0.504|0.000|||||
|250002378|412449921|15.269|638|0|9728.86|0|9729.29|0|9730.37|0.667|0.000|||||
|922564322|412436139|15.260|169|0|233.89|0|234.42|0|235.97|0.488|0.008|||||
|205106000|258262000|15.255|2722|0|1893.07|0|1893.22|0|1893.97|0.627|0.000|||||

Notes: overlap/validation rows come from the validated “top‑100” overlap artifacts, so they may be blank for pairs outside that set.
Flag/gear columns are filled only when this pair appears in the Aug 2017 enrichment CSV (undirected match); most full‑coverage top pairs will have these blank.
