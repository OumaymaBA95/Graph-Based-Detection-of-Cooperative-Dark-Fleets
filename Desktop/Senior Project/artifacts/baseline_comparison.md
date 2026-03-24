# Baseline Comparison

Summary:

Linear GAE baselines outperform heuristic methods on ROC AUC while achieving similar average precision. Prototype year-specific runs are lower as expected due to fewer edges, but provide a validated end-to-end pipeline for year-sliced experiments.

Recent runs:

|timestamp|model|edges|test_ratio|roc_auc|average_precision|use_features|features_years|
|---|---|---|---|---|---|---|---|
|2026-02-18T11:32:32.959372Z|tgcn_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_full.parquet|0.3|0.6786916343773299|0.6954778674873502|True|tgcn_time+temporal_node|
|2026-02-18T02:54:03.662235Z|tgcn_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.8832195304425624|0.9214496206771222|True|tgcn_time+temporal_node|
|2026-02-18T02:31:21.496409Z|tgcn_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.7840000000000001|0.8781746031746032|True|tgcn_time+temporal_node|
|2026-02-18T02:14:40.667394Z|gconvlstm_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.4776666666666666|0.7019100529100529|True|gconvlstm_time|
|2026-02-18T02:06:41.532086Z|gconvgru_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.4144166666666666|0.6639087301587302|True|gconvgru_time|
|2026-02-18T01:58:31.264947Z|tgcn_time_multiseed|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.51175|0.710989010989011|True|tgcn_time|
|2026-02-18T01:51:47.216174Z|tgcn|/Users/momoba/Desktop/Senior Project/artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.78125|0.8791666666666667|True|tgcn|
|2026-02-17T20:28:53.631648Z|tgn_lite|artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.3318785955167625|0.4620860531479798|True|tgn_lite|
|2026-02-17T20:26:58.927683Z|tgn_lite|artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.0|0.0|True|tgn_lite|
|2026-02-17T20:20:07.903952Z|edge_temporal|artifacts/edges_2012_2019_cap5000_even30.parquet|0.3|0.5|0.5|True|edge_temporal|
