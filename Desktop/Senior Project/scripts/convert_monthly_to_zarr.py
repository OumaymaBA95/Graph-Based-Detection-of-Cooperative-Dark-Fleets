#!/usr/bin/env python3
"""Convert monthly GLORYS NetCDF files to a consolidated Zarr store.

Usage examples:
  # convert all months found under data/ into one Zarr
  python3 scripts/convert_monthly_to_zarr.py --input-dir data --pattern "mercatorglorys12v1_gl12_mean_*.nc" --out data/glorys_2012_2019.zarr

  # convert only first N files (test)
  python3 scripts/convert_monthly_to_zarr.py --input-dir data --pattern "mercatorglorys12v1_gl12_mean_*.nc" --out data/glorys_test_q1.zarr --n-files 3

This script uses xarray.open_mfdataset and writes a consolidated Zarr. Adjust chunks to match your memory.
"""
import argparse
from pathlib import Path
import xarray as xr
import multiprocessing


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--pattern', default='mercatorglorys12v1_gl12_mean_*.nc')
    p.add_argument('--out', required=True)
    p.add_argument('--vars', nargs='+', default=['thetao'], help='Variables to keep (default: thetao)')
    p.add_argument('--n-files', type=int, default=None, help='If set, only use the first N matching files (useful for testing)')
    p.add_argument('--depth-index', type=int, default=0, help='Depth index to select (if depth dimension present)')
    p.add_argument('--chunks', default="{'time':1,'latitude':512,'longitude':512}", help='dask chunks dict, as Python literal')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit('No files matched pattern in ' + str(input_dir))
    if args.n_files is not None:
        files = files[:args.n_files]
    print('Found', len(files), 'files. First:', files[0])

    chunks = eval(args.chunks)

    # open multiple files as one dataset
    print('Opening files with xarray.open_mfdataset (this may take a while)...')
    ds = xr.open_mfdataset([str(p) for p in files], combine='by_coords', parallel=False, chunks=chunks)

    # select variables of interest if present
    available_vars = [v for v in args.vars if v in ds.variables]
    if not available_vars:
        # try common names
        for cand in ('thetao','tos','sst'):
            if cand in ds.variables:
                available_vars = [cand]
                break
    if not available_vars:
        raise SystemExit('No requested variables found in dataset. Available vars: ' + ','.join(list(ds.data_vars)))

    ds_sel = ds[available_vars]

    # If depth exists, select the depth index (surface)
    if 'depth' in ds_sel.dims:
        try:
            ds_sel = ds_sel.isel(depth=args.depth_index)
        except Exception:
            # ignore if not possible
            pass

    out_path = Path(args.out)
    if out_path.exists():
        print('Output exists; will overwrite:', out_path)

    print('Writing Zarr to', out_path)
    # Use consolidated metadata for efficient opening
    ds_sel.to_zarr(str(out_path), mode='w', consolidated=True)
    print('Done')


if __name__ == '__main__':
    main()
