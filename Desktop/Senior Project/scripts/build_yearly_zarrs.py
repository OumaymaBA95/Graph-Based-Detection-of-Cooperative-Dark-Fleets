#!/usr/bin/env python3
"""Build per-year Zarr stores from monthly GLORYS NetCDF files already placed in `data/`.

Usage:
  python3 scripts/build_yearly_zarrs.py --input-dir data --out-dir data/glorys_by_year --start-year 2012 --end-year 2019

The script finds files matching the pattern 'mercatorglorys12v1_gl12_mean_{YYYYMM}.nc' for each year and
uses xarray.open_mfdataset to combine them by coordinates, selects the requested variable (default 'thetao') and
surface depth (depth index 0), then writes a consolidated Zarr per year.

Notes:
- This opens one year's worth of monthly files at a time to keep memory usage reasonable.
- You can specify a bounding box to reduce output size.
"""
import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import os


def process_year(year, input_dir, out_dir, varnames, depth_index, chunks, bbox):
    print(f"\nProcessing year {year}...")
    pattern = f"mercatorglorys12v1_gl12_mean_{year:04d}*.nc"
    files = sorted(Path(input_dir).glob(pattern))
    if not files:
        print(f"  No files found for year {year} with pattern {pattern}")
        return False
    print(f"  Found {len(files)} files for {year}, first: {files[0].name}")

    try:
        ds = xr.open_mfdataset([str(p) for p in files], combine='by_coords', parallel=False, chunks=chunks)
    except Exception as e:
        print('  Failed to open files for year', year, e)
        return False

    # Select variables
    available = [v for v in varnames if v in ds.variables]
    if not available:
        for cand in ('thetao', 'tos', 'sst'):
            if cand in ds.variables:
                available = [cand]
                break
    if not available:
        print('  No suitable variables found for year', year, 'available:', list(ds.data_vars))
        ds.close()
        return False

    ds_sel = ds[available]

    # select bbox if provided
    if bbox is not None:
        lonmin, lonmax, latmin, latmax = bbox
        # longitude coords in file are 'longitude' or 'lon' -> handle both
        lon_name = None
        for name in ('longitude','lon'):
            if name in ds_sel.coords:
                lon_name = name
                break
        lat_name = None
        for name in ('latitude','lat'):
            if name in ds_sel.coords:
                lat_name = name
                break
        if lon_name and lat_name:
            ds_sel = ds_sel.sel({lon_name: slice(lonmin, lonmax), lat_name: slice(latmin, latmax)})
        else:
            print('  Could not find lat/lon coords to apply bbox; continuing without bbox')

    # if depth present, select depth index
    if 'depth' in ds_sel.dims:
        try:
            ds_sel = ds_sel.isel(depth=depth_index)
        except Exception:
            pass

    out_path = Path(out_dir) / f'glorys_{year}.zarr'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print('  Output already exists, will overwrite:', out_path)

    try:
        print('  Writing Zarr to', out_path)
        ds_sel.to_zarr(str(out_path), mode='w', consolidated=True)
        print('  Done writing', out_path)
    except Exception as e:
        print('  Failed to write zarr for year', year, e)
        ds.close()
        return False

    ds.close()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--start-year', type=int, required=True)
    p.add_argument('--end-year', type=int, required=True)
    p.add_argument('--vars', nargs='+', default=['thetao'], help='Variables to keep')
    p.add_argument('--depth-index', type=int, default=0)
    p.add_argument('--chunks', default="{'time':1,'latitude':512,'longitude':512}")
    p.add_argument('--bbox', default=None, help='Optional bbox lonmin,lonmax,latmin,latmax')
    args = p.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    chunks = eval(args.chunks)
    bbox = None
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(',')]
        if len(parts) == 4:
            bbox = parts

    for year in range(args.start_year, args.end_year + 1):
        ok = process_year(year, input_dir, out_dir, args.vars, args.depth_index, chunks, bbox)
        if not ok:
            print(f'Processing year {year} failed or had missing files; see messages above')

    print('\nAll done')

if __name__ == '__main__':
    main()
