"""
Build human-readable vessel-pair labels: country (flag) + gear from enrichment CSV,
with ITU MID → short country code fallback when enrichment is missing.

Used by plot_cooperative_heatmap.py, plot_cooperative_timeline.py, overlap_by_month_8pairs.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mmsi_mid import mmsi_to_mid

# ITU MID (first 3 digits) → ISO 3166-1 alpha-3 style tag (excerpt; unknown MIDs → literal MID###).
_MID_COUNTRY: dict[int, str] = {
    201: "ALB",
    224: "ESP",
    225: "ESP",
    226: "ESP",
    227: "FRA",
    228: "FRA",
    229: "MLT",
    230: "FIN",
    231: "FRO",
    232: "GBR",
    233: "GBR",
    234: "GBR",
    235: "GBR",
    238: "NOR",
    240: "LVA",
    244: "FIN",
    245: "NLD",
    246: "NLD",
    247: "ITA",
    248: "MLT",
    249: "CYP",
    250: "IRL",
    251: "HRV",
    252: "SVN",
    255: "PRT",
    256: "MLT",
    257: "NOR",
    258: "DNK",
    259: "NOR",
    261: "POL",
    263: "DEU",
    265: "SWE",
    266: "BEL",
    268: "BGR",
    271: "ROU",
    272: "UKR",
    273: "RUS",
    274: "RUS",
    275: "LVA",
    276: "EST",
    277: "LTU",
    301: "AUS",
    303: "USA",
    304: "ATG",
    306: "NLD",
    307: "NLD",
    308: "NLD",
    310: "BMU",
    311: "BHS",
    319: "CYM",
    320: "GRC",
    321: "GRC",
    338: "USA",
    339: "JAM",
    345: "MEX",
    346: "MEX",
    366: "USA",
    367: "USA",
    368: "USA",
    369: "USA",
    370: "PAN",
    371: "PAN",
    372: "PAN",
    373: "PAN",
    374: "PAN",
    375: "VEN",
    401: "AFG",
    403: "SAU",
    408: "BHR",
    410: "PAK",
    411: "IND",
    412: "CHN",
    413: "CHN",
    414: "CHN",
    416: "JOR",
    422: "IRN",
    423: "IRN",
    424: "ARE",
    431: "JPN",
    432: "JPN",
    434: "UZB",
    440: "KOR",
    441: "JPN",
    443: "ISR",
    445: "PRK",
    446: "KOR",
    447: "KWT",
    452: "BGD",
    457: "LAO",
    458: "VNM",
    459: "VNM",
    460: "CHN",
    461: "CHN",
    466: "TWN",
    469: "PHL",
    472: "LKA",
    477: "HKG",
    503: "AUS",
    512: "NZL",
    525: "IDN",
    533: "MYS",
    566: "SGP",
    567: "THA",
    601: "SYC",
    603: "AGO",
    605: "DZA",
    609: "MUS",
    610: "SEN",
    615: "NGA",
    616: "CMR",
    623: "EGY",
    626: "KEN",
    627: "TZA",
    629: "MDG",
    636: "LBR",
    637: "CAF",
    638: "SDN",
    639: "UGA",
    641: "SOM",
    645: "MOZ",
    650: "MWI",
    654: "ZMB",
    656: "ZAF",
    657: "ZAF",
    659: "ZAF",
    663: "MRT",
    665: "CPV",
    667: "ERI",
    701: "ARG",
    710: "BRA",
    720: "BRA",
    730: "CHL",
    735: "COL",
    740: "ECU",
    745: "ECU",
    750: "ARG",
    759: "PRY",
    770: "URY",
    800: "IDN",
    801: "IDN",
    810: "PHL",
    811: "PHL",
    820: "THA",
    821: "THA",
    822: "THA",
    823: "THA",
    824: "THA",
    825: "THA",
    826: "THA",
    827: "THA",
    828: "THA",
    829: "THA",
    830: "THA",
    831: "THA",
    832: "THA",
    833: "THA",
    834: "THA",
    835: "THA",
    836: "THA",
    837: "THA",
    838: "THA",
    839: "THA",
    840: "THA",
    841: "THA",
    842: "THA",
    843: "THA",
    844: "THA",
    845: "THA",
    846: "THA",
    847: "THA",
    848: "THA",
    849: "THA",
    850: "THA",
    851: "THA",
    852: "THA",
    853: "THA",
    854: "THA",
    855: "THA",
    856: "THA",
    857: "THA",
    858: "THA",
    859: "THA",
    860: "THA",
    861: "THA",
    862: "THA",
    863: "THA",
    864: "THA",
    865: "THA",
    866: "THA",
    867: "THA",
    868: "THA",
    869: "THA",
    870: "IDN",
    871: "IDN",
    872: "IDN",
    873: "IDN",
    874: "IDN",
    875: "IDN",
    876: "IDN",
    877: "IDN",
    878: "IDN",
    879: "IDN",
    880: "KOR",
    881: "KOR",
    882: "KOR",
    883: "KOR",
    884: "KOR",
    885: "THA",
    886: "THA",
    887: "THA",
    888: "THA",
    889: "THA",
    890: "THA",
    891: "THA",
    892: "THA",
    893: "THA",
    894: "THA",
    895: "THA",
    896: "THA",
    897: "THA",
    898: "THA",
    899: "THA",
    # PRC / SAR blocks often used in AIS for Chinese vessels (fleet-specific; treat as CHN for display)
    900: "CHN",
    901: "CHN",
    902: "CHN",
    903: "CHN",
    904: "CHN",
    905: "CHN",
    906: "CHN",
    907: "CHN",
    908: "CHN",
    909: "CHN",
    910: "CHN",
    911: "CHN",
    912: "CHN",
    913: "CHN",
    914: "CHN",
    915: "CHN",
    916: "CHN",
    917: "CHN",
    918: "CHN",
    919: "CHN",
    920: "CHN",
    921: "CHN",
    922: "CHN",
    923: "CHN",
    924: "CHN",
    925: "CHN",
    926: "CHN",
    927: "CHN",
    928: "CHN",
    929: "CHN",
    930: "CHN",
    931: "CHN",
    932: "CHN",
    933: "CHN",
    934: "CHN",
    935: "CHN",
    936: "CHN",
    937: "CHN",
    938: "CHN",
    939: "CHN",
    940: "CHN",
    941: "CHN",
    942: "CHN",
    943: "CHN",
    944: "CHN",
    945: "CHN",
    946: "CHN",
    947: "CHN",
    948: "CHN",
    949: "CHN",
    950: "CHN",
    951: "CHN",
    952: "CHN",
    953: "CHN",
    954: "CHN",
    955: "CHN",
    956: "CHN",
    957: "CHN",
    958: "CHN",
    959: "CHN",
    960: "CHN",
    961: "CHN",
    962: "CHN",
    963: "CHN",
    964: "CHN",
    965: "CHN",
    966: "CHN",
    967: "CHN",
    968: "CHN",
    969: "CHN",
    970: "CHN",
    971: "CHN",
    972: "CHN",
    973: "CHN",
    974: "CHN",
    975: "CHN",
    976: "CHN",
    977: "CHN",
    978: "CHN",
    979: "CHN",
    980: "CHN",
    981: "CHN",
    982: "CHN",
    983: "CHN",
    984: "CHN",
    985: "CHN",
    986: "CHN",
    987: "CHN",
    988: "CHN",
    989: "CHN",
    990: "CHN",
    991: "CHN",
    992: "CHN",
    993: "CHN",
    994: "CHN",
    995: "CHN",
    996: "CHN",
    997: "CHN",
    998: "CHN",
    999: "CHN",
}


def mid_to_country(mid: int) -> str:
    return _MID_COUNTRY.get(int(mid), f"MID{int(mid):03d}")


def country_for_mmsi(mmsi: int) -> str:
    return mid_to_country(mmsi_to_mid(int(mmsi)))


def _clean(s: str | float | int | None) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip()
    return t if t else ""


def load_enrichment_first_row_per_pair(path: Path | str) -> dict[tuple[int, int], dict[str, str]]:
    """Undirected key (min, max) -> first row dict with flag/gears."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "src" not in df.columns or "dst" not in df.columns:
        return {}
    out: dict[tuple[int, int], dict[str, str]] = {}
    for _, r in df.iterrows():
        a, b = int(r["src"]), int(r["dst"])
        key = (a, b) if a <= b else (b, a)
        if key in out:
            continue
        out[key] = {
            "src_flag_cell": _clean(r.get("src_flag_cell", "")),
            "dst_flag_cell": _clean(r.get("dst_flag_cell", "")),
            "src_gear": _clean(r.get("src_gear", "")),
            "dst_gear": _clean(r.get("dst_gear", "")),
        }
    return out


def vessel_country(mmsi: int, flag_cell: str) -> str:
    """Prefer grid-cell flag (e.g. CHN); else ITU MID mapping."""
    f = _clean(flag_cell)
    if f:
        u = f.upper()
        if u.startswith("UNKNOWN"):
            return country_for_mmsi(mmsi)
        return f.split("-")[0] if "-" in f else f
    return country_for_mmsi(mmsi)


def load_mmsi_country_gear_map(enrich_path: Path | str) -> dict[int, tuple[str, str]]:
    """
    MMSI -> (country, gear) from enrichment CSV (first occurrence per MMSI).
    Vessels not in the file must be resolved with country_for_mmsi + '—' for gear.
    """
    p = Path(enrich_path)
    out: dict[int, tuple[str, str]] = {}
    if not p.exists():
        return out
    df = pd.read_csv(p)
    if "src" not in df.columns:
        return out
    for _, r in df.iterrows():
        for side in ("src", "dst"):
            col_m = side
            m = int(r[col_m])
            if m in out:
                continue
            fc = _clean(r.get(f"{side}_flag_cell", ""))
            gr = _clean(r.get(f"{side}_gear", ""))
            c = vessel_country(m, fc)
            g = gr if gr else "—"
            out[m] = (c, g)
    return out


def mmsi_country_gear_line(mmsi: int, mmsi_map: dict[int, tuple[str, str]]) -> str:
    """Short label: country · gear (fallback ITU + —)."""
    c, g = mmsi_map.get(mmsi, (country_for_mmsi(mmsi), "—"))
    return f"{c} · {g}"


def format_gear_display(gear: str | None) -> str:
    """
    Human-readable gear for titles (cooperative CSV uses snake_case).
    Empty / unknown → em dash.
    """
    g = _clean(gear)
    if not g or g == "—":
        return "—"
    return g.replace("_", " ").strip()


def vessel_country_gear_resolved(
    mmsi: int,
    side: str,
    row: dict[str, str] | None,
    mmsi_map: dict[int, tuple[str, str]],
) -> tuple[str, str]:
    """
    Country + gear for one vessel in a pair.

    Precedence: same-row flag/gear from pair enrichment, then any MMSI-level gear
    from ``mmsi_map`` (other rows), then ITU MID + —.
    """
    if row:
        fc = row.get(f"{side}_flag_cell", "")
        gr = row.get(f"{side}_gear", "")
        c = vessel_country(mmsi, fc)
        if gr:
            g = format_gear_display(gr)
        elif mmsi in mmsi_map:
            _, g2 = mmsi_map[mmsi]
            g = format_gear_display(g2) if g2 and g2 != "—" else "—"
        else:
            g = "—"
        return c, g
    if mmsi in mmsi_map:
        c, g2 = mmsi_map[mmsi]
        g = format_gear_display(g2) if g2 and g2 != "—" else "—"
        return c, g
    return country_for_mmsi(mmsi), "—"


def pair_plot_title_country_gear(
    src: int,
    dst: int,
    enrich_pairs: dict[tuple[int, int], dict[str, str]] | None,
    mmsi_map: dict[int, tuple[str, str]],
) -> tuple[str, str | None]:
    """
    Two-line title block: main metadata line + optional short provenance.

    Uses undirected pair key for ``enrich_pairs``. Gear strings are prettified.
    """
    key = (src, dst) if src <= dst else (dst, src)
    row = enrich_pairs.get(key) if enrich_pairs else None
    c1, g1 = vessel_country_gear_resolved(src, "src", row, mmsi_map)
    c2, g2 = vessel_country_gear_resolved(dst, "dst", row, mmsi_map)
    main = f"{src}: {c1} · {g1}    │    {dst}: {c2} · {g2}"
    prov: str | None = None
    if g1 == "—" and g2 == "—":
        prov = "ITU country from MMSI prefix; gear not in cooperative-pairs CSV for these vessels."
    return main, prov


def format_pair_label(
    src: int,
    dst: int,
    enrich: dict[tuple[int, int], dict[str, str]] | None,
    *,
    multiline: bool = True,
) -> str:
    """
    MMSIs + per-vessel country + gear.
    If enrichment missing for this pair, uses ITU MID for country and em dash for gear.
    """
    key = (src, dst) if src <= dst else (dst, src)
    row = enrich.get(key) if enrich else None
    if row:
        sf = vessel_country(src, row["src_flag_cell"])
        df = vessel_country(dst, row["dst_flag_cell"])
        sg = row["src_gear"] or "—"
        dg = row["dst_gear"] or "—"
    else:
        sf = country_for_mmsi(src)
        df = country_for_mmsi(dst)
        sg, dg = "—", "—"

    pair_id = f"{src}–{dst}"
    if multiline:
        return f"{pair_id}\n{sf}·{sg}  |  {df}·{dg}"
    return f"{pair_id}  {sf}·{sg} | {df}·{dg}"
