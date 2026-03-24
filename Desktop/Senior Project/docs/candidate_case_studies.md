# Candidate Pair Case Studies (Validation‑Focused, Full Coverage)

These case studies focus on pairs that (a) score highly in the full‑coverage candidate ranking and (b) also have **non‑zero close‑proximity days** under at least one validation threshold. This helps separate “same broad region” overlap from more direct co‑movement signals.

Sources:
- Scores: `artifacts/tgcn_candidate_scores_fullcoverage.parquet` (1500-bucket model, ROC AUC 0.78)
- Scores (legacy): `artifacts/tgcn_candidate_scores_fullcoverage_top500.csv`
- Close‑proximity validation tables:
  - `artifacts/close_pairs_25km_w1.csv`
  - `artifacts/close_pairs_50km_w3.csv`
  - `artifacts/close_pairs_100km_w7.csv`
  - `artifacts/close_pairs_fullcoverage_25km_w1.csv` (new 1500-bucket model, 100 files/year)

## New validated pairs (1500-bucket model, top-100)

From `artifacts/close_pairs_fullcoverage_25km_w1.csv` — 8 pairs pass 25km ±1 day:

| src | dst | days_within_25km | mean_distance_km |
|-----|-----|------------------|------------------|
| 412422375 | 412428225 | **2** | 15.3 |
| 412437423 | 412435485 | 1 | 10.1 |
| 412410128 | 412416248 | 1 | 13.3 |
| 412461376 | 412415321 | 1 | 138.6 |
| 412420679 | 412413383 | 1 | 78.0 |
| 412985698 | 412443375 | 1 | 348.5 |
| 412450427 | 111203412 | 1 | 137.4 |
| 412461376 | 412427825 | 1 | 191.1 |

*Plots: `artifacts/plots/case_study_pairs_fullcoverage/`*

## Summary table (original 4, recommended)

|src|dst|score|within25km_±1d|within50km_±3d|within100km_±7d|plot|
|---|---|---|---|---|---|---|
|412000690|412325200|15.9186|102|410|1568|[pair_412000690_412325200_contour.png](../artifacts/plots/case_study_pairs/pair_412000690_412325200_contour.png) (tracks-only: [`pair_412000690_412325200.png`](../artifacts/plots/case_study_pairs/pair_412000690_412325200.png))|
|412061791|412508302|15.8086|8|88|378|[pair_412061791_412508302.png](../artifacts/plots/case_study_pairs/pair_412061791_412508302.png)|
|412425192|998508450|14.8642|9|20|53|[pair_412425192_998508450.png](../artifacts/plots/case_study_pairs/pair_412425192_998508450.png)|
|978925333|415131223|14.8048|2|12|864|—|

*Plots show daily mean lat/lon tracks for each vessel; generated via `scripts/compute_pair_overlap_from_daily.py`. Use **`--contour`** for dual filled density overlays (same style as Figure 1 in `docs/final_report.md`).*

## How to interpret these
- `within*` counts days where the two vessels were within the threshold (km) allowing the time window (±days). Higher counts under the tighter thresholds (25km ±1d) are strongest.
- A pair can still be interesting even if it only appears at 50km/100km windows, but treat those as weaker evidence.

## Six-vessel cluster near 30°N, 122°E

Beyond individual pairs, we asked whether the high-scoring candidates reveal a small group of vessels that repeatedly meet in the same area. Using the monthly 25 km overlap file (`artifacts/eight_pairs_overlap_by_month.csv`), we identified a six-vessel cluster (MMSIs 412461376, 412413383, 412420679, 412427825, 412435485, and 412437423) whose rendezvous locations concentrate in a narrow region around 30°N, 122°E. Each of these vessels accumulates 17–24 days of close proximity with at least one partner over 2013–2016, and their inferred rendezvous points lie within roughly half a degree of latitude and longitude. This pattern suggests a coordinated operating area—consistent with a small fleet or fishing–support network—rather than isolated chance encounters.

The plot `artifacts/plots/six_vessel_cluster_scatter.png` visualizes all inferred rendezvous points (background) and highlights those involving these six vessels, along with the hotspot centroid.

## Pair highlights

### 412000690 ↔ 412325200
- **Strongest validation signal:** 102 days within 25km ±1 day; 410 days within 50km ±3 days; 1,568 days within 100km ±7 days.
- Mean distance when both vessels are active: ~135 km (25km window) to ~139 km (100km window), indicating sustained co‑presence in the same operating area.
- Overlap of 774 days (25km) and 1,767 days (50km) suggests multi‑year, repeated close encounters rather than a single incident.
- **Interpretation:** Highest‑confidence candidate for cooperative or coordinated behavior; warrants external metadata lookup (flag, gear, AIS history) for validation.
- **Use in write‑up:** Primary case study for “model surfaces plausible bridge‑vessel pairs.”

![Pair 412000690 and 412325200 (dual filled contours + tracks)](../artifacts/plots/case_study_pairs/pair_412000690_412325200_contour.png)

### 412061791 ↔ 412508302
- **Moderate validation:** 8 days within 25km ±1 day; 88 days within 50km ±3 days; 378 days within 100km ±7 days.
- Mean distance ~216 km (25km) to ~252 km (100km)—tighter at 25km, looser at wider windows.
- Lower overlap (226 days at 25km) than pair 1; suggests intermittent rather than persistent close proximity.
- **Interpretation:** Possible transshipment rendezvous or shared fishing grounds; weaker evidence than pair 1 but still above chance.
- **Use in write‑up:** Secondary case study showing the model can surface pairs with fewer but non‑zero close‑proximity days.

![Pair 412061791 and 412508302](../artifacts/plots/case_study_pairs/pair_412061791_412508302.png)

### 412425192 ↔ 998508450
- **Small but non‑zero validation:** 9 days within 25km ±1 day; 20 days within 50km ±3 days; 53 days within 100km ±7 days.
- Mean distance ~194 km (25km) to ~189 km (100km).
- Overlap of 140 days (25km) and 355 days (50km)—fewer total encounters than pairs 1 and 2.
- **Interpretation:** Could indicate occasional coordination or shared port/region; useful for geographic diversity in the case‑study set.
- **Use in write‑up:** Tertiary example; demonstrates that the model ranks pairs across different overlap/ proximity profiles.

![Pair 412425192 and 998508450](../artifacts/plots/case_study_pairs/pair_412425192_998508450.png)

### 978925333 ↔ 415131223
- **Weak at 25km, strong at 100km:** 2 days within 25km ±1 day; 12 days within 50km ±3 days; 864 days within 100km ±7 days.
- Mean distance ~140 km (25km) to ~139 km (100km)—consistent across windows.
- High overlap (537 days at 25km, 1,218 at 50km) but very few close‑proximity days at tight thresholds.
- **Interpretation:** Likely regional co‑activity (same fishing grounds or migration corridor) rather than direct side‑by‑side coordination; treat as “broad co‑movement” evidence.
- **Use in write‑up:** Contrast case—shows the model can surface regionally aligned pairs that do *not* have strong close‑proximity validation, illustrating the value of multi‑threshold checks.

![Pair 978925333 and 415131223](../artifacts/plots/case_study_pairs/pair_978925333_415131223.png)

---

## External Metadata Lookup (Template)

**How to look up vessels:**
1. **MarineTraffic** (marinetraffic.com): Search by MMSI; shows flag, type, dimensions, recent positions.
2. **Global Fishing Watch** (globalfishingwatch.org): Vessel profiles, gear type, flag.
3. **IHS Markit / Equasis:** Commercial registries; more detailed for some vessels.

To strengthen case-study interpretation, look up each MMSI and fill in:

| MMSI | Flag | Vessel type / gear | Length (m) | Notes |
|------|------|--------------------|------------|-------|
| 412000690 | | | | |
| 412325200 | | | | |
| 412061791 | | | | |
| 412508302 | | | | |
| 412425192 | | | | |
| 998508450 | | | | |
| 978925333 | | | | |
| 415131223 | | | | |
| 412422375 | | | | | *(strongest new pair, 2 days within 25km)* |
| 412428225 | | | | |

*412xxxxx = China; 998xxxxx = Indonesia; 978xxxxx = Thailand; 415xxxxx = Taiwan.*

---

## Limitations

- **Grid‑cell resolution:** Overlap and distance are computed from daily grid‑cell positions. Fine‑scale transshipment (e.g., vessels meeting within a few km for hours) may be blurred or missed.
- **No ground truth:** We have no labeled “cooperative” or “bridge‑vessel” pairs; validation is based on proximity heuristics, not confirmed behavior.
- **MMSI validity:** A small number of candidate IDs (e.g., under 100M) may be non‑standard; we filter these in reporting but they can appear in raw outputs.
- **Temporal coverage:** Results depend on the 2012–2019 slice and the specific edge‑construction parameters (distance threshold, time bucket, sampling).
