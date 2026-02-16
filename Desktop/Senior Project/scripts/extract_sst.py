#!/usr/bin/env python3
"""Extract SST values at vessel positions using OceanBench (or synthetic fallback).

Writes a Parquet with columns: MMSI, timestamp, lat, lon, sst, sst_missing

Usage (quick test):
  python3 scripts/extract_sst.py --input data/combined_fleet_daily.csv --output data/sst_points_sample.parquet --sample 200
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import requests
import os
import time


def try_import_oceanbench():
    try:
        import oceanbench
        return oceanbench
    except Exception:
        return None


def find_column(df, candidates):
    """Find a column name in a dataframe or coordinate-like object.

    Accepts a pandas DataFrame/Index or an xarray Coordinates mapping. Matching is
    case-insensitive and will return the first candidate found.
    """
    cols = None
    # xarray coords can be accessed like a mapping
    try:
        # pandas Index or list-like
        cols = list(df.columns)
    except Exception:
        try:
            cols = list(df)
        except Exception:
            cols = []

    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c is None:
            continue
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def normalize_lon_array(lon_array, ds_lon):
    # ds_lon is a 1D array-like of dataset longitudes
    lonmin = float(np.nanmin(ds_lon))
    lonmax = float(np.nanmax(ds_lon))
    lon = lon_array.copy()
    if lonmin >= 0 and lonmax > 180:
        # dataset uses 0..360
        lon = np.where(lon < 0, lon + 360, lon)
    else:
        # dataset likely -180..180
        lon = np.where(lon > 180, lon - 360, lon)
    return lon


def _get_matrix_dimensions(tilematrix_id: str):
    """Return matrix width/height for EPSG:4326 TileMatrix with id as integer string.

    This mirrors the pattern in the WMTS capabilities where MatrixWidth = 2^(z+1)
    and MatrixHeight = 2^z for the provided capability document.
    """
    z = int(tilematrix_id)
    matrix_width = 2 ** (z + 1)
    matrix_height = 2 ** z
    return matrix_width, matrix_height


def _lonlat_to_tile(lon, lat, matrix_width, matrix_height):
    # Capabilities EPSG:4326 TopLeftCorner = (90, -180)
    top_left_lat = 90.0
    left_lon = -180.0
    tile_deg_lon = 360.0 / matrix_width
    tile_deg_lat = 180.0 / matrix_height
    col = int(np.floor((lon - left_lon) / tile_deg_lon))
    row = int(np.floor((top_left_lat - lat) / tile_deg_lat))
    col = max(0, min(matrix_width - 1, col))
    row = max(0, min(matrix_height - 1, row))
    return col, row, tile_deg_lon, tile_deg_lat


def _pixel_in_tile(lon, lat, col, row, tile_deg_lon, tile_deg_lat, tile_width_px=256, tile_height_px=256):
    tile_min_lon = -180.0 + col * tile_deg_lon
    tile_max_lat = 90.0 - row * tile_deg_lat
    x_frac = (lon - tile_min_lon) / tile_deg_lon
    y_frac = (tile_max_lat - lat) / tile_deg_lat
    i = int(round(x_frac * (tile_width_px - 1)))
    j = int(round(y_frac * (tile_height_px - 1)))
    i = max(0, min(tile_width_px - 1, i))
    j = max(0, min(tile_height_px - 1, j))
    return i, j


def _find_first_numeric(obj):
    """Recursively find the first numeric (int/float) value in a nested JSON-like object."""
    if obj is None:
        return None
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            r = _find_first_numeric(v)
            if r is not None:
                return r
    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = _find_first_numeric(v)
            if r is not None:
                return r
    return None


def extract_sst_with_wmts(df, time_col, lat_col, lon_col, mmsi_col=None, sample_limit=None, tilematrix='6', pause=0.1):
    """Query Copernicus WMTS GetFeatureInfo per point and return a DataFrame with sst values.

    This is a fallback for small batches. It performs one HTTP request per point and
    attempts to find the first numeric value in the JSON response.
    """
    WMTS_BASE = "https://wmts.marine.copernicus.eu/teroWmts"
    LAYER = "GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/thetao"
    TILEMATRIXSET = "EPSG:4326"

    records = df.to_dict('records')
    if sample_limit is not None:
        records = records[:sample_limit]

    mmsi_field = mmsi_col or find_column(df, ['MMSI', 'mmsi', 'mmsi_id', 'mmsi_present']) or (list(df.columns)[0] if len(df.columns) else None)

    session = requests.Session()
    # support basic auth via env vars (CMEMS)
    user = os.environ.get('CMEMS_USER')
    pwd = os.environ.get('CMEMS_PASS')
    if user and pwd:
        session.auth = (user, pwd)

    matrix_w, matrix_h = _get_matrix_dimensions(tilematrix)

    out_rows = []
    for rec in records:
        t = rec.get(time_col)
        lat = float(rec.get(lat_col))
        lon = float(rec.get(lon_col))
        # ensure ISO8601 Z suffix
        try:
            t_iso = pd.to_datetime(t).strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            t_iso = str(t)

        col, row, td_lon, td_lat = _lonlat_to_tile(lon, lat, matrix_w, matrix_h)
        i, j = _pixel_in_tile(lon, lat, col, row, td_lon, td_lat)

        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetFeatureInfo',
            'VERSION': '1.0.0',
            'LAYER': LAYER,
            'STYLE': '',
            'FORMAT': 'image/png',
            'TileMatrixSet': TILEMATRIXSET,
            'TileMatrix': tilematrix,
            'TileCol': str(col),
            'TileRow': str(row),
            'I': str(i),
            'J': str(j),
            'INFOFORMAT': 'application/json',
            'TIME': t_iso,
        }

        sst = np.nan
        try:
            r = session.get(WMTS_BASE, params=params, timeout=20)
            r.raise_for_status()
            try:
                payload = r.json()
                val = _find_first_numeric(payload)
                if val is not None:
                    sst = float(val)
            except ValueError:
                # not JSON — ignore
                sst = np.nan
        except Exception:
            sst = np.nan

        out_rows.append((rec.get(mmsi_field), t, lat, lon, sst))
        time.sleep(pause)

    out_df = pd.DataFrame(out_rows, columns=['MMSI', 'timestamp', 'lat', 'lon', 'sst'])
    out_df['sst_missing'] = out_df['sst'].isna()
    return out_df


def extract_sst_with_oceanbench(oceanbench, df, time_col, lat_col, lon_col, mmsi_col=None, sample_limit=None):
    import xarray as xr
    ds = None
    # Some oceanbench installs include a glorys helper under oceanbench.core.references
    try:
        from oceanbench.core.references import glorys as glorys_helper
    except Exception:
        glorys_helper = None

    if glorys_helper is None:
        raise RuntimeError('OceanBench glorys helper not found in this environment')

    # We'll open per-day Zarr stores (the helper provides a URL builder) to avoid loading huge multi-day datasets.
    # Group sample rows by date (YYYYMMDD) and open the corresponding Zarr store for interpolation.
    import xarray as xr
    df_local = df if sample_limit is None else df.iloc[:sample_limit]
    # Ensure time column is datetime
    df_local[time_col] = pd.to_datetime(df_local[time_col])
    df_local['day'] = df_local[time_col].dt.floor('D')

    out_rows = []
    # determine mmsi field name
    mmsi_field = mmsi_col or find_column(df, ['MMSI', 'mmsi', 'mmsi_id', 'mmsi_present']) or (list(df.columns)[0] if len(df.columns) else None)

    for day, sub in df_local.groupby('day'):
        try:
            url = glorys_helper._glorys_1_4_path(np.datetime64(day))
        except Exception as e:
            print('Could not build GLORYS URL for day', day, e)
            for rec in sub.to_dict('records'):
                out_rows.append((rec.get(mmsi_field), rec.get(time_col), float(rec.get(lat_col)), float(rec.get(lon_col)), np.nan))
            continue

        try:
            print('Opening GLORYS Zarr:', url)
            ds_day = xr.open_zarr(url)
        except Exception as e:
            print('Failed to open zarr at', url, e)
            for rec in sub.to_dict('records'):
                out_rows.append((rec.get(mmsi_field), rec.get(time_col), float(rec.get(lat_col)), float(rec.get(lon_col)), np.nan))
            continue

        # pick sst-like var
        var = None
        for cand in ('thetao', 'sst', 'tos'):
            if cand in ds_day:
                var = cand
                break
        if var is None:
            var = list(ds_day.data_vars)[0]

        da = ds_day[var]
        if 'depth' in da.dims:
            try:
                depth_vals = np.array(ds_day.coords['depth'])
                idx = int(np.argmin(np.abs(depth_vals - 0)))
                da = da.isel(depth=idx)
            except Exception:
                da = da.isel(depth=0)

        # coordinate names
        lon_name = find_column(ds_day.coords, ['longitude', 'lon', 'long']) or list(ds_day.coords)[0]
        lat_name = find_column(ds_day.coords, ['latitude', 'lat']) or list(ds_day.coords)[1]
        time_name = find_column(ds_day.coords, ['time']) or list(ds_day.coords)[2]

        ds_lon = ds_day.coords[lon_name].values if lon_name in ds_day.coords else None

        for rec in sub.to_dict('records'):
            t = rec.get(time_col)
            lat = float(rec.get(lat_col))
            lon = float(rec.get(lon_col))
            if ds_lon is not None:
                lon = normalize_lon_array(np.array([lon]), ds_lon)[0]
            try:
                v = da.interp({time_name: np.datetime64(t), lat_name: float(lat), lon_name: float(lon)})
                sst = float(v.values)
            except Exception:
                sst = np.nan
            out_rows.append((rec.get(mmsi_field), t, lat, lon, sst))

    out_df = pd.DataFrame(out_rows, columns=['MMSI', 'timestamp', 'lat', 'lon', 'sst'])
    out_df['sst_missing'] = out_df['sst'].isna()
    return out_df

    # find SST-like variable
    sst_var = None
    for cand in ('thetao', 'sst', 'tos'):
        if cand in ds:
            sst_var = cand
            break
    if sst_var is None:
        sst_var = list(ds.data_vars)[0]
        print("Using first data var as SST-like:", sst_var)

    da = ds[sst_var]
    # If depth present, select nearest-surface
    if 'depth' in da.dims:
        try:
            depth_vals = np.array(ds.coords['depth'])
            idx = int(np.argmin(np.abs(depth_vals - 0)))
            da = da.isel(depth=idx)
            print(f"Selected depth index {idx} for surface (depth={depth_vals[idx]})")
        except Exception:
            da = da.isel(depth=0)

    # find coordinate names
    lon_name = None
    lat_name = None
    time_name = None
    for name in ('longitude', 'lon', 'long'):
        if name in ds.coords:
            lon_name = name
            break
    for name in ('latitude', 'lat'):
        if name in ds.coords:
            lat_name = name
            break
    for name in ('time',):
        if name in ds.coords:
            time_name = name
            break

    ds_lon = ds.coords[lon_name].values if lon_name else None

    # Prepare output columns
    out_rows = []
    it = df.itertuples(index=False)
    if sample_limit is not None:
        it = (row for i, row in enumerate(df.itertuples(index=False)) if i < sample_limit)

    for row in it:
        t = getattr(row, time_col)
        lat = float(getattr(row, lat_col))
        lon = float(getattr(row, lon_col))
        if ds_lon is not None:
            lon = normalize_lon_array(np.array([lon]), ds_lon)[0]
        try:
            # scalar interpolation
            val = da.interp({time_name: np.datetime64(t), lat_name: float(lat), lon_name: float(lon)})
            sst = float(val.values)
        except Exception:
            sst = np.nan
        out_rows.append((getattr(row, find_column(df, ['MMSI', 'mmsi', 'mmsi_id']) or 'MMSI'), t, lat, lon, sst))

    out_df = pd.DataFrame(out_rows, columns=['MMSI', 'timestamp', 'lat', 'lon', 'sst'])
    out_df['sst_missing'] = out_df['sst'].isna()
    return out_df



def extract_sst_from_xarray(ds, df, time_col, lat_col, lon_col, mmsi_col=None, sample_limit=None, time_tolerance_days: int = 16):
    """Interpolate SST from a provided xarray Dataset `ds` for rows in df."""
    import xarray as xr
    # Identify a likely SST variable
    sst_var = None
    for cand in ('thetao', 'sst', 'tos'):
        if cand in ds:
            sst_var = cand
            break
    if sst_var is None:
        sst_var = list(ds.data_vars)[0]
        print('Using first data var as SST-like:', sst_var)

    da = ds[sst_var]
    # If depth present, select surface
    if 'depth' in da.dims:
        try:
            depth_vals = np.array(ds.coords['depth'])
            idx = int(np.argmin(np.abs(depth_vals - 0)))
            da = da.isel(depth=idx)
            print(f"Selected depth index {idx} for surface (depth={depth_vals[idx]})")
        except Exception:
            da = da.isel(depth=0)

    # coord names
    lon_name = None
    lat_name = None
    time_name = None
    for name in ('longitude', 'lon', 'long'):
        if name in ds.coords:
            lon_name = name
            break
    for name in ('latitude', 'lat'):
        if name in ds.coords:
            lat_name = name
            break
    for name in ('time',):
        if name in ds.coords:
            time_name = name
            break

    ds_lon = ds.coords[lon_name].values if lon_name else None
    # gather time values for nearest-time selection
    time_vals = None
    if time_name and time_name in ds.coords:
        try:
            time_vals = np.array(ds.coords[time_name].values)
        except Exception:
            time_vals = None

    out_rows = []
    records = df.to_dict('records')
    if sample_limit is not None:
        records = records[:sample_limit]

    mmsi_field = mmsi_col or find_column(df, ['MMSI', 'mmsi', 'mmsi_id', 'mmsi_present']) or (list(df.columns)[0] if len(df.columns) else None)

    for rec in records:
        t = rec.get(time_col)
        lat = float(rec.get(lat_col))
        lon = float(rec.get(lon_col))
        if ds_lon is not None:
            lon = normalize_lon_array(np.array([lon]), ds_lon)[0]
        sst = np.nan
        try:
            # If the dataset has a time coordinate, pick nearest time within tolerance
            if time_vals is not None and len(time_vals) > 0:
                t64 = np.datetime64(t)
                # compute absolute deltas
                deltas = np.abs(time_vals - t64)
                # convert tolerance to numpy timedelta
                tol = np.timedelta64(int(time_tolerance_days), 'D')
                idx = int(deltas.argmin())
                if deltas[idx] <= tol:
                    # select the time slice then interpolate over lat/lon
                    try:
                        da_time = da.isel({time_name: idx})
                        # try bilinear/spatial interpolation first
                        try:
                            val = da_time.interp({lat_name: float(lat), lon_name: float(lon)})
                            sst = float(val.values)
                        except Exception:
                            sst = np.nan
                        # if interpolation yielded NaN, try nearest-grid fallback
                        if np.isnan(sst):
                            try:
                                val2 = da_time.sel({lat_name: float(lat), lon_name: float(lon)}, method='nearest')
                                sst = float(val2.values)
                            except Exception:
                                sst = np.nan
                    except Exception:
                        sst = np.nan
                else:
                    # no nearby time within tolerance
                    sst = np.nan
            else:
                # fallback: full interpolation including time
                try:
                    val = da.interp({time_name: np.datetime64(t), lat_name: float(lat), lon_name: float(lon)})
                    sst = float(val.values)
                except Exception:
                    sst = np.nan
                if np.isnan(sst):
                    # try nearest-grid (ignoring time interpolation) as a last resort
                    try:
                        # select nearest time then nearest grid point
                        if time_name in ds.coords:
                            # find nearest time index
                            tvals = np.array(ds.coords[time_name].values)
                            idx2 = int(np.abs(tvals - np.datetime64(t)).argmin())
                            da_time2 = da.isel({time_name: idx2})
                            val2 = da_time2.sel({lat_name: float(lat), lon_name: float(lon)}, method='nearest')
                            sst = float(val2.values)
                        else:
                            val2 = da.sel({lat_name: float(lat), lon_name: float(lon)}, method='nearest')
                            sst = float(val2.values)
                    except Exception:
                        sst = np.nan
        except Exception:
            sst = np.nan
        out_rows.append((rec.get(mmsi_field), t, lat, lon, sst))

    out_df = pd.DataFrame(out_rows, columns=['MMSI', 'timestamp', 'lat', 'lon', 'sst'])
    out_df['sst_missing'] = out_df['sst'].isna()
    return out_df


def extract_sst_from_netcdf(nc_path, df, time_col, lat_col, lon_col, mmsi_col=None, sample_limit=None):
    """Extract SST using netCDF4 by reading small windows per point to avoid loading full arrays.

    This function expects a variable like 'thetao' (temperature) or another 3/4D var
    with dims (time, depth, lat, lon) or (time, lat, lon). It will select the surface
    depth if present and perform bilinear interpolation from the four nearest grid cells.
    """
    from netCDF4 import Dataset
    import math

    ds = Dataset(nc_path, 'r')
    # find sst-like variable
    varname = None
    for cand in ('thetao', 'sst', 'tos'):
        if cand in ds.variables:
            varname = cand
            break
    if varname is None:
        # pick first variable with 3 or 4 dims
        for name, v in ds.variables.items():
            if len(v.dimensions) >= 2:
                varname = name
                break
    if varname is None:
        raise RuntimeError('No suitable variable found in netCDF')

    var = ds.variables[varname]
    dims = var.dimensions
    # locate dim names
    has_depth = 'depth' in dims
    time_dim = None
    lat_dim = None
    lon_dim = None
    for d in dims:
        if 'time' in d:
            time_dim = d
        if d in ('latitude', 'lat'):
            lat_dim = d
        if d in ('longitude', 'lon'):
            lon_dim = d

    # fallback guesses
    if lat_dim is None:
        lat_dim = [d for d in ds.dimensions if 'lat' in d or 'latitude' in d][0]
    if lon_dim is None:
        lon_dim = [d for d in ds.dimensions if 'lon' in d or 'longitude' in d][0]

    lons = ds.variables[lon_dim][:]
    lats = ds.variables[lat_dim][:]

    # choose time index 0 (these are single-day files typically)
    time_idx = 0
    depth_idx = 0
    if has_depth:
        # try to find depth index nearest to 0
        try:
            depth_vals = ds.variables['depth'][:]
            depth_idx = int(np.argmin(np.abs(depth_vals - 0)))
        except Exception:
            depth_idx = 0

    # helper: bilinear interpolation
    def bilinear(pt_lat, pt_lon):
        # find nearest lon/lat indices
        i = np.searchsorted(lons, pt_lon) - 1
        j = np.searchsorted(lats, pt_lat) - 1
        i = max(0, min(i, len(lons) - 2))
        j = max(0, min(j, len(lats) - 2))
        # corners
        lon0, lon1 = float(lons[i]), float(lons[i+1])
        lat0, lat1 = float(lats[j]), float(lats[j+1])
        try:
            if has_depth and len(var.shape) == 4:
                vals = var[time_idx, depth_idx, j:j+2, i:i+2]
            elif len(var.shape) == 3:
                vals = var[time_idx, j:j+2, i:i+2]
            else:
                # unexpected shape: read full 2D
                vals = var[j:j+2, i:i+2]
        except Exception:
            return np.nan
        # ensure floats
        try:
            q11 = float(vals[0,0])
            q12 = float(vals[0,1])
            q21 = float(vals[1,0])
            q22 = float(vals[1,1])
        except Exception:
            return np.nan
        # avoid division by zero
        if lon1 == lon0 or lat1 == lat0:
            return float((q11 + q12 + q21 + q22) / 4.0)
        # interpolation weights
        tx = (pt_lon - lon0) / (lon1 - lon0)
        ty = (pt_lat - lat0) / (lat1 - lat0)
        a = q11 * (1 - tx) + q12 * tx
        b = q21 * (1 - tx) + q22 * tx
        return float(a * (1 - ty) + b * ty)

    records = df.to_dict('records')
    if sample_limit is not None:
        records = records[:sample_limit]

    mmsi_field = mmsi_col or find_column(df, ['MMSI', 'mmsi', 'mmsi_id', 'mmsi_present']) or (list(df.columns)[0] if len(df.columns) else None)
    out_rows = []
    for rec in records:
        try:
            t = rec.get(time_col)
            lat = float(rec.get(lat_col))
            lon = float(rec.get(lon_col))
            # normalize lon into dataset range
            if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
                lon = lon if lon >= 0 else lon + 360
            else:
                lon = lon if lon <= 180 else lon - 360
            sst = bilinear(lat, lon)
        except Exception:
            sst = np.nan
        out_rows.append((rec.get(mmsi_field), rec.get(time_col), lat, lon, sst))

    out_df = pd.DataFrame(out_rows, columns=['MMSI', 'timestamp', 'lat', 'lon', 'sst'])
    out_df['sst_missing'] = out_df['sst'].isna()
    ds.close()
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to combined fleet CSV')
    p.add_argument('--output', required=True, help='Path to output Parquet')
    p.add_argument('--time-col', default=None, help='Column name for timestamp (overrides auto-detect)')
    p.add_argument('--lat-col', default=None, help='Column name for latitude (overrides auto-detect)')
    p.add_argument('--lon-col', default=None, help='Column name for longitude (overrides auto-detect)')
    p.add_argument('--mmsi-col', default=None, help='Column name for MMSI (overrides auto-detect)')
    p.add_argument('--ocean-ds', default=None, help='Path to local NetCDF/Zarr dataset for SST (optional)')
    p.add_argument('--sample', type=int, default=None, help='If set, only process this many rows')
    p.add_argument('--chunksize', type=int, default=10000, help='Chunk size for CSV reading')
    p.add_argument('--use-wmts', action='store_true', help='Use WMTS GetFeatureInfo fallback for small batches')
    p.add_argument('--force-synthetic', action='store_true', help='Use synthetic SST instead of oceanbench')
    p.add_argument('--time-tolerance-days', type=int, default=16, help='Nearest-time tolerance (days) when using monthly-mean products')
    args = p.parse_args()

    oceanbench = None
    if not args.force_synthetic:
        oceanbench = try_import_oceanbench()
        if oceanbench is None:
            print('oceanbench not available in environment; to use OceanBench install it with: pip install oceanbench')
    else:
        print('force synthetic SST enabled (note: synthetic SST generation has been removed)')

    in_path = Path(args.input)
    if not in_path.exists():
        print('Input file not found:', in_path)
        sys.exit(1)

    # Read a small sample (if requested) for quick test
    df = pd.read_csv(in_path, parse_dates=True)

    # Detect common column names or use overrides
    time_col = args.time_col or find_column(df, ['timestamp', 'time', 'datetime', 'date'])
    lat_col = args.lat_col or find_column(df, ['lat', 'latitude', 'cell_ll_lat'])
    lon_col = args.lon_col or find_column(df, ['lon', 'longitude', 'long', 'cell_ll_lon'])
    mmsi_col = args.mmsi_col or find_column(df, ['MMSI', 'mmsi', 'mmsi_id', 'mmsi_present', 'mmsi'])

    if not all([time_col, lat_col, lon_col]):
        print('Could not detect required columns. Found columns:', df.columns.tolist())
        print('Please pass explicit column names with --time-col --lat-col --lon-col --mmsi-col')
        sys.exit(1)

    # Ensure timestamp parsed
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col, lat_col, lon_col])

    sample_limit = args.sample
    if args.sample is not None:
        print(f'Processing sample of {args.sample} rows')

    out_df = None
    # Prefer OceanBench dataset helpers if present
    if oceanbench is not None and hasattr(oceanbench, 'datasets'):
        try:
            out_df = extract_sst_with_oceanbench(oceanbench, df, time_col, lat_col, lon_col, mmsi_col=mmsi_col, sample_limit=sample_limit)
        except Exception as e:
            print('OceanBench extraction failed:', e)
    # If the user supplied a local dataset path, try opening with xarray
    if out_df is None and args.ocean_ds is not None:
        import xarray as xr
        try:
            print('Opening local dataset with xarray:', args.ocean_ds)
            ds_path = args.ocean_ds
            try:
                pth = Path(ds_path)
                is_zarr = pth.is_dir() or str(ds_path).endswith('.zarr')
            except Exception:
                is_zarr = str(ds_path).endswith('.zarr')

            if is_zarr:
                # open zarr lazily with conservative chunks to avoid memory spikes
                ds = xr.open_zarr(ds_path, chunks={'time': 1, 'latitude': 512, 'longitude': 512})
            else:
                ds = xr.open_dataset(ds_path, chunks={'time': 1, 'latitude': 512, 'longitude': 512})

            out_df = extract_sst_from_xarray(ds, df, time_col, lat_col, lon_col, mmsi_col=mmsi_col, sample_limit=sample_limit, time_tolerance_days=args.time_tolerance_days)
        except MemoryError as me:
            print('Out of memory while opening dataset:', me)
            sys.exit(3)
        except Exception as e:
            print('Failed to open or extract from local dataset:', e)

    if out_df is None:
        # Try WMTS fallback if user requested it (suitable for small batches)
        if args.use_wmts:
            try:
                print('Attempting WMTS GetFeatureInfo fallback (small-batch)')
                out_df = extract_sst_with_wmts(df, time_col, lat_col, lon_col, mmsi_col=mmsi_col, sample_limit=sample_limit)
            except Exception as e:
                print('WMTS extraction failed:', e)

    if out_df is None:
        print('ERROR: Could not extract SST from OceanBench, WMTS, or the provided local dataset.')
        print('Please provide a valid --ocean-ds path or install OceanBench, or try --use-wmts for small tests with CMEMS credentials in CMEMS_USER/CMEMS_PASS.')
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, engine='pyarrow', index=False)
    print('Wrote', out_path, 'rows=', len(out_df))


if __name__ == '__main__':
    main()
