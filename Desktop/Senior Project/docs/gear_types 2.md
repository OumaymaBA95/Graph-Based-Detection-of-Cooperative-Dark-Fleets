# Gear type labels (enrichment CSV)

**Thesis / final report:** The **canonical** text for a bound PDF is **`docs/final_report.md` Appendix A** (gear table + interpretation). This file is a **mirror** for easy editing in the repo—**keep Appendix A in sync** when you change definitions here (or vice versa). **Appendix C** in the same report holds **notation** and the formal link-prediction setup.

§4.3.1 in the main report summarizes; **Appendix A** has the full table (gear appears in **Tables 2–3** in the report body).

In this project, **gear** on figures and tables comes from **cell-level fleet aggregates**, not from a per-vessel registry. For each MMSI–day, we join AIS daily cells to fleet files where each row is **hours** by **flag × geartype** in that cell; the vessel is labeled with the **geartype that receives the most weighted hours** that day (`scripts/enrich_pairs_with_flag_gear.py`). So the label is a **coarse, behavioral proxy** for “what kind of fishing that cell’s traffic was attributed to,” not a formal gear certificate for the MMSI.

**Plots use `—`** when no gear could be attributed (vessel missing from the join, or no overlapping fleet rows).

---

## Codes used in `artifacts/cooperative_pairs_with_flag_gear.csv`

These strings are taken **as-is** from the fleet `geartype` column (snake_case). Below is a plain-language gloss so readers know what each code *usually* refers to in fisheries terminology.

| Code (CSV) | Short gloss |
|------------|-------------|
| **`fishing`** | **Generic / unspecified fishing.** Catch-all category in the source data when activity is attributed to “fishing” without a more specific gear class. Treat as **low specificity** compared to trawlers, nets, etc. |
| **`trawlers`** | **Trawling.** Vessels **tow nets** through the water column or along the bottom (e.g. otter trawl, beam trawl). High mobility; distinct from **fixed** or **set** gear. |
| **`fixed_gear`** | **Stationary gear** fixed to the seabed or structure: pots, traps, stakes, weirs, and similar **non-towed** gear that stays in place. (Name reflects “fixed” position, not “repaired.”) |
| **`set_gillnets`** | **Set gillnets.** Nets **set and left** (anchored, on bottom, or sometimes drifting) so fish **gill** in the mesh. Not actively towed like a trawl. |
| **`set_longlines`** | **Set longlines.** A **long line** with many **baited hooks**, deployed and **left to soak** (demersal or pelagic longline, depending on fishery). |
| **`other_purse_seines`** | **Purse seining (other).** A **surrounding net** used on **schooling fish**; the bottom is **pulled closed** (“pursed”) like a drawstring. “Other” indicates a sub-type bucket in the source taxonomy (not necessarily “miscellaneous quality”). |
| **`pole_and_line`** | **Pole-and-line / baitboat.** Fish caught with **hand-held poles** and **hooks**, often with **live bait**; common in some tuna fisheries. |

---

## How to interpret labels in your writeup

1. **Same MMSI, different days** can get different geartypes if the vessel moves between cells dominated by different fleet attributions.
2. **Same geartype on two MMSIs** does not prove they use identical gear—only that both were attributed the same **coarse** class in that **August 2017** window.
3. **Comparison to AIS “ship type”** is different: fleet **geartype** here is **fisheries-activity** oriented, not the full IMO/AIS ship-type code list.

For methodology of the join and weighting, see `scripts/enrich_pairs_with_flag_gear.py` and `docs/final_report.md` (flag / gear sections).

---

## Related (not from the same CSV)

The repo also has a **movement-based gear classifier** trained on **anonymized** labeled tracks (`README.md` → “Gear classification”). Those class labels (e.g. `purse_seines`, `trollers`) are **not** automatically the same strings as the fleet `geartype` column above; do not merge them without an explicit mapping.
