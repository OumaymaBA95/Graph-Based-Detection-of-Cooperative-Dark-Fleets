#!/usr/bin/env python3
"""Vectorized SST extractor for per-year CSVs against a per-year Zarr.

This script reads a per-year CSV, groups sample rows by nearest dataset time index,
and for each time-slice performs a vectorized xarray interpolation (lat/lon arrays)
followed by a nearest-grid fallback for NaNs. Outputs parquet chunk files matching
previous layout.

Usage:
  python3 scripts/extract_sst_vectorized.py --input data/by_year/combined_2013.csv --zarr data/glorys_by_year/glorys_2013.zarr --out-dir data/sst_by_year/2013

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import xarray as xr


def find_column(df, candidates):
    # df can be a pandas DataFrame, an xarray Coordinates mapping, or list-like
    cols = None
    try:
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


def vectorized_extract(zarr_path, csv_path, out_dir, chunk_size=50000, time_tolerance_days=31):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Opening dataset', zarr_path)
    ds = xr.open_zarr(str(zarr_path), chunks={'time':1,'latitude':512,'longitude':512})
    # select sst-like var
    sst_var = None
    for cand in ('thetao','sst','tos'):
        if cand in ds:
            sst_var = cand; break
    if sst_var is None:
        sst_var = list(ds.data_vars)[0]
    da = ds[sst_var]
    # depth handling
    if 'depth' in da.dims:
        try:
            depth_vals = np.array(ds.coords['depth'])
            idx = int(np.argmin(np.abs(depth_vals - 0)))
            da = da.isel(depth=idx)
        except Exception:
            da = da.isel(depth=0)
    # coords
    lon_name = find_column(ds.coords, ['longitude','lon','long']) or list(ds.coords)[0]
    lat_name = find_column(ds.coords, ['latitude','lat']) or list(ds.coords)[1]
    time_name = 'time' if 'time' in ds.coords else list(ds.coords)[0]
    time_vals = np.array(ds.coords[time_name].values)
    lons_ds = np.array(ds.coords[lon_name].values)

    # read CSV in chunks to limit memory
    preview = pd.read_csv(csv_path, nrows=20)
    time_col = find_column(preview, ['timestamp','time','datetime','date']) or preview.columns[0]
    lat_col = find_column(preview, ['lat','latitude','cell_ll_lat'])
    lon_col = find_column(preview, ['lon','longitude','cell_ll_lon'])
    mmsi_col = find_column(preview, ['MMSI','mmsi','mmsi_id','mmsi_present'])
    if not all([time_col, lat_col, lon_col]):
        raise RuntimeError('Could not detect time/lat/lon columns in csv')

    reader = pd.read_csv(csv_path, parse_dates=[time_col], chunksize=chunk_size)
    chunk_i = 0
    for chunk in reader:
        print('Processing csv chunk', chunk_i, 'rows', len(chunk))
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors='coerce')
        chunk = chunk.dropna(subset=[time_col, lat_col, lon_col])
        if chunk.empty:
            chunk_i += 1
            continue
        # normalize longitudes to dataset domain
        lonmin = float(np.nanmin(lons_ds))
        lonmax = float(np.nanmax(lons_ds))
        lons = chunk[lon_col].astype(float).to_numpy()
        if lonmin >= 0 and lonmax > 180:
            lons = np.where(lons < 0, lons + 360, lons)
        else:
            lons = np.where(lons > 180, lons - 360, lons)
        lats = chunk[lat_col].astype(float).to_numpy()
        times = chunk[time_col].to_numpy().astype('datetime64[ns]')
        # compute nearest time index per row
        deltas = np.abs(time_vals.reshape((1,-1)) - times.reshape((-1,1)))
        idxs = deltas.argmin(axis=1)
        # apply tolerance
        tol = np.timedelta64(int(time_tolerance_days), 'D')
        keep_mask = np.min(deltas, axis=1) <= tol
        # prepare output array
        sst_out = np.full(len(chunk), np.nan, dtype=float)
        # group rows by idxs where keep_mask True
        import collections
        groups = collections.defaultdict(list)
        for i, (keep, idx) in enumerate(zip(keep_mask, idxs)):
            if keep:
                groups[int(idx)].append(i)
        # For each time index, vectorized interp
        for tidx, rows_idx in groups.items():
            da_time = da.isel({time_name: int(tidx)})
            # perform vectorized interpolation with xarray
            try:
                arr = da_time.interp({lat_name: xr.DataArray(lats[rows_idx]), lon_name: xr.DataArray(lons[rows_idx])})
                vals = np.array(arr)
                # where NaN, attempt nearest grid selection
                nanpos = np.where(np.isnan(vals))[0]
                if len(nanpos) > 0:
                    try:
                        sel = da_time.sel({lat_name: xr.DataArray(lats[rows_idx][nanpos]), lon_name: xr.DataArray(lons[rows_idx][nanpos])}, method='nearest')
                        vals[nanpos] = np.array(sel)
                    except Exception:
                        pass
                sst_out[rows_idx] = vals
            except Exception:
                # fallback to per-point nearest
                for j in rows_idx:
                    try:
                        val = da_time.sel({lat_name: float(lats[j]), lon_name: float(lons[j])}, method='nearest')
                        sst_out[j] = float(val.values)
                    except Exception:
                        sst_out[j] = np.nan
        # write results to parquet with same columns pattern
        out_df = pd.DataFrame({
            'MMSI': chunk[mmsi_col] if mmsi_col in chunk.columns else None,
            'timestamp': chunk[time_col],
            'lat': chunk[lat_col].astype(float),
            'lon': chunk[lon_col].astype(float),
            'sst': sst_out
        })
        out_df['sst_missing'] = out_df['sst'].isna()
        out_file = Path(out_dir) / f'chunk_{chunk_i*chunk_size}.parquet'
        out_df.to_parquet(out_file, engine='pyarrow', index=False)
        print('Wrote', out_file, 'rows', len(out_df))
        chunk_i += 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--zarr', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--chunk-size', type=int, default=50000)
    p.add_argument('--time-tolerance-days', type=int, default=31)
    args = p.parse_args()
    vectorized_extract(args.zarr, args.input, args.out_dir, chunk_size=args.chunk_size, time_tolerance_days=args.time_tolerance_days)
