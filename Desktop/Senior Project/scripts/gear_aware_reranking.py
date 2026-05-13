#!/usr/bin/env python3
"""
Gear-aware post-hoc re-ranking of TGCN candidate pairs.

Three-step pipeline:
  1. Enrich top-N candidate MMSIs with dominant gear type (sampled fleet join).
  2. Compute gear-pair co-location baseline from fleet daily data.
  3. Re-rank candidates by gear-adjusted score and produce a comparison figure.

Outputs:
  artifacts/candidate_gear_enrichment.csv   -- per-MMSI gear labels
  artifacts/gear_pair_baseline_rates.csv    -- co-location rates per gear combo
  artifacts/candidate_gear_reranked.csv     -- full re-ranked list
  artifacts/plots/gear_aware_reranking.png  -- stacked-bar comparison figure
"""
from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

from mmsi_mid import mmsi_to_mid

_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

MID_COUNTRY = {
    201: "ALB", 211: "DEU", 212: "CYP", 215: "MLT", 219: "DNK",
    224: "ESP", 225: "ESP", 226: "FRA", 227: "FRA", 228: "FRA",
    230: "FIN", 231: "FIN", 232: "GBR", 233: "GBR", 235: "GBR",
    236: "GBR", 237: "GRE", 240: "GRC", 241: "GRC", 244: "NLD",
    245: "NLD", 246: "NLD", 247: "ITA", 249: "MLT", 250: "IRL",
    255: "PRT", 256: "MLT", 257: "NOR", 258: "NOR", 259: "NOR",
    261: "POL", 263: "PRT", 265: "SWE", 266: "SWE", 269: "CZE",
    271: "TUR", 272: "UKR", 273: "RUS", 274: "RUS", 275: "LVA",
    276: "EST", 277: "LTU",
    301: "AIA", 303: "USA", 304: "ATG", 305: "ATG", 306: "CUW",
    307: "ARU", 308: "BHS", 309: "BHS", 311: "BHS", 312: "BLZ",
    314: "BRB", 316: "BMU", 319: "CYM", 321: "CRI", 323: "CUB",
    325: "DMA", 327: "DOM", 329: "GLP", 330: "GRD", 331: "GLP",
    332: "GTM", 334: "HND", 336: "HTI", 338: "USA", 339: "JAM",
    341: "KNA", 343: "LCA", 345: "MEX", 347: "MTQ", 348: "MSR",
    350: "NIC", 351: "PAN", 352: "PAN", 353: "PAN", 354: "PAN",
    355: "PAN", 356: "PAN", 357: "PAN", 358: "PRI", 359: "SLV",
    361: "SPM", 362: "TTO", 364: "TCA", 366: "USA", 367: "USA",
    368: "USA", 369: "USA", 370: "PAN", 371: "PAN", 372: "PAN",
    373: "PAN", 374: "PAN", 375: "VCT", 376: "VCT", 377: "VCT",
    378: "VGB", 379: "VIR",
    401: "AFG", 403: "SAU", 405: "BGD", 408: "BHR", 410: "BTN",
    412: "CHN", 413: "CHN", 414: "CHN", 416: "TWN", 417: "LKA",
    419: "IND", 422: "IRN", 423: "AZE", 425: "IRQ", 428: "ISR",
    431: "JPN", 432: "JPN", 434: "TKM", 436: "KAZ", 437: "UZB",
    438: "JOR", 440: "KOR", 441: "KOR", 443: "PSE", 445: "PRK",
    447: "KWT", 450: "LBN", 451: "KGZ", 453: "MAC", 455: "MDV",
    457: "MNG", 459: "NPL", 461: "OMN", 463: "PAK", 466: "QAT",
    468: "SYR", 470: "ARE", 472: "TJK", 473: "YEM", 475: "YEM",
    477: "HKG", 478: "BIH", 501: "ADE", 503: "AUS",
    506: "MMR", 508: "BRN", 510: "FSM", 511: "PLW", 512: "NZL",
    514: "KHM", 515: "KHM", 516: "CXR", 518: "COK", 520: "FJI",
    523: "CCK", 525: "IDN", 529: "KIR", 531: "LAO", 533: "MYS",
    536: "MHL", 538: "MHL", 540: "NCL", 542: "NIU", 544: "NRU",
    546: "PYF", 548: "PHL", 553: "PNG", 555: "PCN", 557: "SLB",
    559: "WSM", 561: "WSM", 563: "SGP", 564: "SGP", 565: "SGP",
    566: "SGP", 567: "THA", 570: "TON", 572: "TUV", 574: "VNM",
    576: "VUT", 577: "VUT", 578: "WLF",
    601: "ZAF", 603: "AGO", 605: "DZA", 607: "FRA", 608: "GBR",
    609: "BDI", 610: "BEN", 611: "BWA", 612: "CMR", 613: "CPV",
    615: "COG", 616: "COM", 617: "COD", 618: "CIV", 619: "CIV",
    620: "DJI", 621: "EGY", 622: "GNQ", 624: "ETH", 625: "ERI",
    626: "GAB", 627: "GMB", 629: "GHA", 630: "GIN", 631: "GNB",
    632: "GIN", 633: "BFA", 634: "KEN", 635: "KEN", 636: "LBR",
    637: "LBR", 638: "LBY", 642: "LBY", 644: "LSO", 645: "MUS",
    647: "MDG", 649: "MLI", 650: "MOZ", 654: "MRT", 655: "MWI",
    656: "NER", 657: "NGA", 659: "NAM", 660: "REU", 661: "RWA",
    662: "STP", 663: "SEN", 664: "SYC", 665: "SLE", 666: "SOM",
    667: "SLE", 668: "SWZ", 669: "SDN", 670: "TZA", 671: "TZA",
    672: "TCD", 674: "TGO", 675: "TUN", 676: "UGA", 677: "TZA",
    678: "ZMB", 679: "ZWE",
    701: "ARG", 710: "BRA", 720: "BOL", 725: "CHL", 730: "COL",
    735: "ECU", 740: "FLK", 745: "GUF", 750: "GUY", 755: "PRY",
    760: "PER", 765: "SUR", 770: "URY", 775: "VEN",
    800: "unknown",
    900: "unknown",
}

GEAR_DISPLAY = {
    "trawlers": "Trawlers",
    "fishing": "Line / generic",
    "set_gillnets": "Set gillnets",
    "set_longlines": "Set longlines",
    "fixed_gear": "Fixed gear",
    "other_purse_seines": "Purse seines",
    "pole_and_line": "Pole & line",
    "squid_jigger": "Squid jigger",
    "drifting_longlines": "Drifting longlines",
    "pots_and_traps": "Pots & traps",
    "tuna_purse_seines": "Tuna purse seines",
    "other_seines": "Other seines",
    "purse_seines": "Purse seines",
    "dredge_fishing": "Dredge fishing",
    "seiners": "Seiners",
    "trollers": "Trollers",
    "unknown": "Unknown",
}


def _mid_to_country(mid: int) -> str:
    return MID_COUNTRY.get(mid, f"MID{mid}")


def _display_gear_name(raw: str) -> str:
    if raw in GEAR_DISPLAY:
        return GEAR_DISPLAY[raw]
    return str(raw).replace("_", " ").title()


def _display_gear_pair(g1: str, g2: str) -> str:
    return " + ".join(sorted([_display_gear_name(g1), _display_gear_name(g2)]))


# ---------------------------------------------------------------------------
# Step 1: Gear Enrichment
# ---------------------------------------------------------------------------

def _sample_dates(years, per_year=12):
    """Return sampled date strings (15th of each month) across years."""
    dates = []
    for y in years:
        for m in range(1, 13):
            if per_year < 12 and m % max(1, 12 // per_year) != 1:
                continue
            day = min(15, calendar.monthrange(y, m)[1])
            dates.append(f"{y}-{m:02d}-{day:02d}")
    return dates


def enrich_mmsis(mmsis, mmsi_daily_root, fleet_daily_root, years, per_year=12):
    """Assign dominant gear type to each MMSI by sampling fleet joins."""
    dates = _sample_dates(years, per_year)
    print(f"  Sampling {len(dates)} dates for gear enrichment...")

    gear_hours = {m: {} for m in mmsis}
    days_seen = {m: 0 for m in mmsis}

    for i, day in enumerate(dates):
        year = day[:4]
        mmsi_path = mmsi_daily_root / f"mmsi-daily-csvs-10-v3-{day}.csv"
        fleet_path = (
            fleet_daily_root
            / f"fleet-daily-csvs-100-v3-{year}"
            / f"fleet-daily-csvs-100-v3-{day}.csv"
        )
        if not mmsi_path.exists() or not fleet_path.exists():
            continue

        mmsi_day = pd.read_csv(
            mmsi_path,
            usecols=["mmsi", "cell_ll_lat", "cell_ll_lon", "hours"],
        )
        mmsi_day["mmsi"] = pd.to_numeric(mmsi_day["mmsi"], errors="coerce")
        mmsi_day = mmsi_day.dropna(subset=["mmsi"])
        mmsi_day["mmsi"] = mmsi_day["mmsi"].astype(int)
        mmsi_day = mmsi_day[mmsi_day["mmsi"].isin(mmsis)]
        if mmsi_day.empty:
            continue

        fleet_day = pd.read_csv(
            fleet_path,
            usecols=["cell_ll_lat", "cell_ll_lon", "geartype", "hours"],
        )

        merged = mmsi_day.merge(
            fleet_day,
            on=["cell_ll_lat", "cell_ll_lon"],
            how="inner",
            suffixes=("_mmsi", "_fleet"),
        )
        if merged.empty:
            continue

        h_col = "hours_mmsi" if "hours_mmsi" in merged.columns else "hours_x"
        merged["_w"] = merged[h_col].fillna(0)

        for m_id in merged["mmsi"].unique():
            sub = merged[merged["mmsi"] == m_id]
            days_seen[m_id] = days_seen.get(m_id, 0) + 1
            for gear, w in sub.groupby("geartype")["_w"].sum().items():
                gear_hours[m_id][gear] = gear_hours[m_id].get(gear, 0.0) + w

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(dates)} sample days...")

    rows = []
    for m_id in sorted(mmsis):
        gh = gear_hours.get(m_id, {})
        if gh:
            best = max(gh, key=gh.get)
            total_w = sum(gh.values())
        else:
            best = "unknown"
            total_w = 0.0
        rows.append({
            "mmsi": m_id,
            "mid": mmsi_to_mid(m_id),
            "country": _mid_to_country(mmsi_to_mid(m_id)),
            "dominant_gear": best,
            "days_observed": days_seen.get(m_id, 0),
            "hours_weighted": round(total_w, 2),
        })

    df = pd.DataFrame(rows)
    print(f"  Enriched {len(df)} MMSIs: "
          f"{(df['dominant_gear'] != 'unknown').sum()} with gear, "
          f"{(df['dominant_gear'] == 'unknown').sum()} unknown")
    return df


# ---------------------------------------------------------------------------
# Step 2: Gear-Pair Co-location Baseline
# ---------------------------------------------------------------------------

def compute_gear_baseline(fleet_daily_root, years, sample_dates=None):
    """
    Compute how often each gear-pair combination co-occurs in the same cell-day.
    """
    if sample_dates is None:
        sample_dates = _sample_dates(years, per_year=4)

    print(f"  Computing baseline from {len(sample_dates)} fleet days...")
    pair_counts = {}
    total_cell_days = 0

    for i, day in enumerate(sample_dates):
        year = day[:4]
        fp = (
            fleet_daily_root
            / f"fleet-daily-csvs-100-v3-{year}"
            / f"fleet-daily-csvs-100-v3-{day}.csv"
        )
        if not fp.exists():
            continue

        fleet = pd.read_csv(fp, usecols=["cell_ll_lat", "cell_ll_lon", "geartype", "hours"])
        fleet = fleet[fleet["hours"] > 0]

        cell_gears = (
            fleet.groupby(["cell_ll_lat", "cell_ll_lon"])["geartype"]
            .apply(set)
            .reset_index()
        )
        total_cell_days += len(cell_gears)

        for _, row in cell_gears.iterrows():
            gears = sorted(row["geartype"])
            for a, b in combinations(gears, 2):
                key = (a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1
            for g in gears:
                key = (g, g)
                pair_counts[key] = pair_counts.get(key, 0) + 1

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(sample_dates)} fleet days...")

    rows = []
    for (a, b), count in sorted(pair_counts.items()):
        rows.append({
            "gear_a": a,
            "gear_b": b,
            "co_occurrence": count,
            "total_cell_days": total_cell_days,
            "baseline_rate": count / total_cell_days if total_cell_days else 0,
        })

    df = pd.DataFrame(rows)
    print(f"  Computed {len(df)} gear-pair baseline rates from "
          f"{total_cell_days} cell-days")
    return df


# ---------------------------------------------------------------------------
# Step 3: Gear-Adjusted Re-ranking
# ---------------------------------------------------------------------------

def _normalize_gear_pair(g1, g2):
    return tuple(sorted([g1, g2]))


def rerank_candidates(candidates, validated, enrichment, baseline):
    """Compute gear-adjusted scores and re-rank."""
    gear_map = dict(zip(enrichment["mmsi"], enrichment["dominant_gear"]))
    country_map = dict(zip(enrichment["mmsi"], enrichment["country"]))

    baseline_lookup = {}
    for _, row in baseline.iterrows():
        key = _normalize_gear_pair(row["gear_a"], row["gear_b"])
        baseline_lookup[key] = row["baseline_rate"]

    max_rate = max(baseline_lookup.values()) if baseline_lookup else 1.0

    val_set = set()
    for _, row in validated.iterrows():
        s, d = int(row["src"]), int(row["dst"])
        dw = row.get("days_within_km", 0)
        if dw > 0:
            val_set.add((min(s, d), max(s, d)))

    rows = []
    for _, row in candidates.iterrows():
        s, d = int(row["src"]), int(row["dst"])
        sg = gear_map.get(s, "unknown")
        dg = gear_map.get(d, "unknown")
        sc = country_map.get(s, "?")
        dc = country_map.get(d, "?")

        pair_key = _normalize_gear_pair(sg, dg)
        rate = baseline_lookup.get(pair_key, 0.0)
        norm_rate = rate / max_rate if max_rate > 0 else 0.0

        if sg == "unknown" or dg == "unknown" or norm_rate == 0:
            discount = 1.0
        else:
            discount = 1.0 / (1.0 + norm_rate * 2)

        adj_score = row["score"] * discount
        canon = (min(s, d), max(s, d))

        gear_label = _display_gear_pair(sg, dg)

        rows.append({
            "src": s,
            "dst": d,
            "tgcn_score": row["score"],
            "src_gear": sg,
            "dst_gear": dg,
            "src_country": sc,
            "dst_country": dc,
            "same_flag": sc == dc,
            "gear_pair": gear_label,
            "baseline_rate": rate,
            "discount": round(discount, 4),
            "gear_adjusted_score": round(adj_score, 4),
            "validated": canon in val_set,
        })

    df = pd.DataFrame(rows)
    df["original_rank"] = df["tgcn_score"].rank(ascending=False, method="min").astype(int)
    df["adjusted_rank"] = df["gear_adjusted_score"].rank(ascending=False, method="min").astype(int)
    df["rank_change"] = df["original_rank"] - df["adjusted_rank"]
    df = df.sort_values("adjusted_rank")
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_GEAR_PALETTE = {
    "Trawlers + Trawlers": "#2563eb",
    "Line / generic + Trawlers": "#7c3aed",
    "Line / generic + Line / generic": "#f59e0b",
    "Set gillnets + Set gillnets": "#10b981",
    "Set gillnets + Trawlers": "#059669",
    "Set longlines + Trawlers": "#dc2626",
    "Set longlines + Set longlines": "#ef4444",
    "Fixed gear + Trawlers": "#6366f1",
    "Fixed gear + Line / generic": "#818cf8",
    "Purse seines + Trawlers": "#ec4899",
    "Pole & line + Trawlers": "#14b8a6",
}

_GEAR_SINGLE_PALETTE = {
    "Trawlers": "#2563eb",
    "Line / generic": "#f59e0b",
    "Set gillnets": "#10b981",
    "Set longlines": "#ef4444",
    "Fixed gear": "#6366f1",
    "Purse seines": "#ec4899",
    "Pole & line": "#14b8a6",
    "unknown": "#9ca3af",
}


def _gear_color(label):
    if label in _GEAR_PALETTE:
        return _GEAR_PALETTE[label]
    if label in _GEAR_SINGLE_PALETTE:
        return _GEAR_SINGLE_PALETTE[label]
    for key in _GEAR_PALETTE:
        if key in label:
            return _GEAR_PALETTE[key]
    if "unknown" in label.lower():
        return "#9ca3af"
    return "#6b7280"


def make_figure(reranked, baseline, out_path):
    """Combined figure answering: why 8 of 200, and should we trust them?"""
    plt.rcParams.update(_PUBLICATION_RC)

    PROJ_ROOT = Path(__file__).resolve().parent.parent

    close = pd.read_csv(PROJ_ROOT / "artifacts" / "close_pairs_fullcoverage_25km_w1.csv")
    close_val = close[close["days_within_km"] > 0].copy()

    n_total = len(reranked)
    n_val = int(reranked["validated"].sum())
    n_failed = n_total - n_val

    summary = (
        reranked.groupby("gear_pair", as_index=False)
        .agg(total=("validated", "size"), passed=("validated", "sum"))
    )
    summary["failed"] = summary["total"] - summary["passed"]
    summary = summary.sort_values("total", ascending=False).reset_index(drop=True)
    n_bars = len(summary)

    # Colors for the 4 gear types that have validated pairs
    GEAR_COLORS = {
        "Trawlers + Trawlers":       "#6366f1",
        "Line / generic + Trawlers": "#f59e0b",
        "Set gillnets + Trawlers":   "#10b981",
        "Fixed gear + Trawlers":     "#059669",
    }
    BAR_BASE = "#94a3b8"
    COLOR_PASS = "#059669"

    # ── Layout ─────────────────────────────────────────────────────────────
    row_h = 0.52
    fig_top_h = max(4.0, 1.8 + n_bars * row_h)
    fig = plt.figure(figsize=(14, fig_top_h + 5.5))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[fig_top_h, 5.0],
        hspace=0.35,
    )
    ax = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL (a): Horizontal bars — total pairs per gear type
    # ═══════════════════════════════════════════════════════════════════════
    y = np.arange(n_bars)[::-1]
    xmax = float(summary["total"].max())

    for i, (_, row) in enumerate(summary.iterrows()):
        tot = int(row["total"])
        pf = int(row["failed"])
        pp = int(row["passed"])
        gear = row["gear_pair"]
        yi = y[i]

        bar_color = GEAR_COLORS.get(gear, BAR_BASE)

        # Full bar = total count (muted tone)
        ax.barh(yi, tot, height=0.55, color=bar_color, alpha=0.25,
                edgecolor="white", linewidth=0.6, zorder=2)

        # Overlay: failed segment (solid muted)
        ax.barh(yi, pf, height=0.55, color=bar_color, alpha=0.50,
                edgecolor="white", linewidth=0.6, zorder=3)

        # Overlay: passed segment (bright green, stacked after failed)
        if pp > 0:
            ax.barh(yi, pp, height=0.55, left=pf, color=COLOR_PASS,
                    edgecolor="white", linewidth=0.6, zorder=4)

        # ── Right-side annotation: "82 total  |  80 failed · 2 passed"
        if pp > 0:
            label = (f"{tot} total   \u2502   {pf} failed  \u00b7  "
                     f"{pp} passed")
        else:
            label = f"{tot} total   \u2502   all {pf} failed"
        ax.text(
            xmax + 1.5, yi, label,
            va="center", ha="left", fontsize=9.5, color="#1e293b",
            zorder=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(summary["gear_pair"], fontsize=10)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0, xmax * 2.35)
    ax.set_xlabel("Number of candidate pairs", fontsize=10.5)
    ax.set_title(
        f"(a)  All {n_total} TGCN candidates by gear type  \u2014  "
        f"{n_failed} failed, {n_val} passed the 25 km geographic filter",
        fontsize=12, fontweight="bold", loc="left", pad=14,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.6)
    ax.set_axisbelow(True)

    # Minimal legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=BAR_BASE, alpha=0.50, label="Failed 25 km filter"),
            Patch(facecolor=COLOR_PASS, label="Passed 25 km filter"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9,
        edgecolor="#e2e8f0",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL (b): The 8 validated pairs table
    # ═══════════════════════════════════════════════════════════════════════
    val_df = reranked[reranked["validated"]].merge(
        close_val[["src", "dst", "days_within_km", "mean_distance_km"]],
        on=["src", "dst"], how="left",
    )
    val_df = val_df.sort_values("days_within_km", ascending=False).reset_index(drop=True)

    ax_bot.axis("off")
    ax_bot.set_title(
        "(b)  The 8 validated pairs  \u2014  each survived an independent geographic check",
        fontsize=12, fontweight="bold", pad=14, loc="left",
    )

    col_labels = ["Pair", "Gear combination", "Flag", "Days \u226425 km",
                  "Mean dist. (km)", "TGCN rank"]
    col_x = [0.00, 0.22, 0.48, 0.60, 0.74, 0.90]
    header_y = 0.90

    for j, (label, x) in enumerate(zip(col_labels, col_x)):
        ax_bot.text(x, header_y, label, fontsize=10, fontweight="bold",
                    color="#374151", va="center",
                    transform=ax_bot.transAxes)
    ax_bot.plot([0.0, 0.98], [header_y - 0.04, header_y - 0.04],
                color="#cbd5e1", linewidth=1.0,
                transform=ax_bot.transAxes, clip_on=False)

    row_spacing = 0.095
    for i, (_, row) in enumerate(val_df.iterrows()):
        row_y = header_y - 0.10 - i * row_spacing
        color = GEAR_COLORS.get(row["gear_pair"], BAR_BASE)
        flag = "same" if row["same_flag"] else "cross"
        pair_label = f"{int(row['src'])} \u2194 {int(row['dst'])}"
        days = int(row["days_within_km"]) if pd.notna(row["days_within_km"]) else "?"
        dist = f"{row['mean_distance_km']:.1f}" if pd.notna(row["mean_distance_km"]) else "?"
        rank = int(row["original_rank"])

        vals = [pair_label, row["gear_pair"], flag, str(days), dist, str(rank)]
        for j, (val, x) in enumerate(zip(vals, col_x)):
            fw = "bold" if j == 1 else "normal"
            c = color if j == 1 else "#1e293b"
            ax_bot.text(x, row_y, val, fontsize=9.5, color=c,
                        fontweight=fw, va="center",
                        transform=ax_bot.transAxes)

        if i % 2 == 0:
            ax_bot.axhspan(
                row_y - 0.035, row_y + 0.035,
                xmin=0.0, xmax=0.98, color="#f8fafc", zorder=0,
                transform=ax_bot.transAxes,
            )

    n_cross = int((val_df["gear_pair"] != "Trawlers + Trawlers").sum())
    ax_bot.text(
        0.0, header_y - 0.10 - len(val_df) * row_spacing - 0.02,
        f"{n_cross} of 8 validated pairs are cross-gear  \u2014  "
        "consistent with genuine vessel-to-vessel encounters, not fleet noise.",
        fontsize=9.5, color="#475569", style="italic", va="center",
        transform=ax_bot.transAxes,
    )

    fig.text(
        0.5, 0.005,
        f"Gear from cell-aggregated fleet daily data (2012\u20132019).",
        ha="center", fontsize=8.5, color="#94a3b8",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",
                    default="artifacts/tgcn_candidates_fullcoverage_top200.csv")
    p.add_argument("--validated",
                    default="artifacts/close_pairs_fullcoverage_25km_w1.csv")
    p.add_argument("--mmsi-daily-root",
                    default="data/MMSI daily vessels ")
    p.add_argument("--fleet-daily-root", default="data")
    p.add_argument("--sample-days-per-year", type=int, default=12)
    p.add_argument("--start-year", type=int, default=2012)
    p.add_argument("--end-year", type=int, default=2020)
    return p.parse_args()


def main():
    args = parse_args()

    candidates = pd.read_csv(args.candidates)
    validated = pd.read_csv(args.validated)
    mmsi_root = Path(args.mmsi_daily_root)
    fleet_root = Path(args.fleet_daily_root)
    years = range(args.start_year, args.end_year)

    all_mmsis = set(candidates["src"].astype(int)) | set(candidates["dst"].astype(int))
    print(f"Top-{len(candidates)} candidates, {len(all_mmsis)} unique MMSIs")

    print("\n=== Step 1: Gear Enrichment ===")
    enrichment = enrich_mmsis(
        all_mmsis, mmsi_root, fleet_root, years, args.sample_days_per_year,
    )
    enrich_out = Path("artifacts/candidate_gear_enrichment.csv")
    enrich_out.parent.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(enrich_out, index=False)
    print(f"  Wrote {enrich_out}")

    print("\n=== Step 2: Gear-Pair Co-location Baseline ===")
    baseline_dates = _sample_dates(years, per_year=4)
    baseline = compute_gear_baseline(fleet_root, years, baseline_dates)
    bl_out = Path("artifacts/gear_pair_baseline_rates.csv")
    baseline.to_csv(bl_out, index=False)
    print(f"  Wrote {bl_out}")

    print("\n=== Step 3: Gear-Adjusted Re-ranking ===")
    reranked = rerank_candidates(candidates, validated, enrichment, baseline)
    rr_out = Path("artifacts/candidate_gear_reranked.csv")
    reranked.to_csv(rr_out, index=False)
    print(f"  Wrote {rr_out}")

    n_val = reranked["validated"].sum()
    n_same_flag = reranked["same_flag"].sum()
    n_total = len(reranked)
    print(f"\n  Summary: {n_val} validated, {n_same_flag}/{n_total} same-flag "
          f"({100*n_same_flag/n_total:.0f}%)")

    gear_dist = reranked["gear_pair"].value_counts()
    print(f"\n  Gear-pair distribution:")
    for g, c in gear_dist.items():
        pct = 100 * c / n_total
        print(f"    {g}: {c} ({pct:.0f}%)")

    val_ranks = reranked[reranked["validated"]][
        ["src", "dst", "original_rank", "adjusted_rank", "rank_change", "gear_pair"]
    ]
    print(f"\n  Validated pair rank shifts:")
    print(val_ranks.to_string(index=False))

    print("\n=== Generating Figure ===")
    fig_out = Path("artifacts/plots/gear_aware_reranking.png")
    make_figure(reranked, baseline, fig_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
