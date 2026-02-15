#!/usr/bin/env python3
"""
WMTS fallback helper (simulate + sample modes).

simulate: build stratified sample plan and example GetFeatureInfo URLs without network calls.
sample: (not implemented in this simulate-first commit) would run real requests with throttling/retries.

This script will write a plan JSON to data/sst_wmts_sample/plan_simulate.json when run in simulate mode.
"""
import argparse
import json
import math
import os
import random
from datetime import datetime

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['simulate', 'sample'], default='simulate')
    p.add_argument('--sample-size', type=int, default=100000)
    p.add_argument('--by-year-dir', default='data/by_year')
    p.add_argument('--out-dir', default='data/sst_wmts_sample')
    p.add_argument('--layer', default='SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/C3S-GLO-SST-L4-REP-OBS-SST_202506/analysed_sst')
    p.add_argument('--tilematrixset', default='EPSG:4326')
    p.add_argument('--tilematrix', type=int, default=4, help='example zoom / TileMatrix level to use for URL examples')
    p.add_argument('--time-sample-day', default='15T00:00:00Z', help='time-of-day used in example TIME param')
    return p.parse_args()


def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)


def latlon_to_tile_webmercator(lat, lon, z, tile_size=256):
    # Use WebMercator formulas for example URLs (EPSG:3857 tiling)
    lat_rad = math.radians(lat)
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    # pixel coordinates within tile
    x_norm = (lon + 180.0) / 360.0 * n - xtile
    # y_norm approximate via mercator
    y_norm = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n - ytile
    pixel_x = int(x_norm * tile_size)
    pixel_y = int(y_norm * tile_size)
    return xtile, ytile, pixel_x, pixel_y


def build_getfeatureinfo_url(base, layer, tilematrixset, tilematrix, tilecol, tilerow, i, j, time):
    # Build a GetFeatureInfo URL template for Copernicus WMTS
    params = (
        f"SERVICE=WMTS&REQUEST=GetFeatureInfo&VERSION=1.0.0"
        f"&LAYER={layer}&STYLE=&TILEMATRIXSET={tilematrixset}"
        f"&TILEMATRIX={tilematrix}&TILEROW={tilerow}&TILECOL={tilecol}"
        f"&I={i}&J={j}&INFOFORMAT=application/json&FORMAT=image/png&TIME={time}"
    )
    if base.endswith('?'):
        return base + params
    if '?' in base:
        return base + '&' + params
    return base + '?' + params


def select_example_points(by_year_dir, years, per_year_examples=10):
    examples = {}
    for y in years:
        csv_path = os.path.join(by_year_dir, f'combined_{y}.csv')
        pts = []
        if os.path.exists(csv_path):
            try:
                # read a small sample from the year CSV
                df = pd.read_csv(csv_path, usecols=['latitude', 'longitude', 'time'], parse_dates=['time'], nrows=10000)
                if df.empty:
                    raise ValueError('empty csv')
                # random sample up to per_year_examples
                df = df.sample(min(per_year_examples, len(df)))
                for _, r in df.iterrows():
                    pts.append({'lat': float(r['latitude']), 'lon': float(r['longitude']), 'time': r['time'].isoformat()})
            except Exception:
                # fallback synthetic points
                for i in range(per_year_examples):
                    pts.append({'lat': -10.0 + i, 'lon': 20.0 + i, 'time': f'{y}-06-15T00:00:00Z'})
        else:
            for i in range(per_year_examples):
                pts.append({'lat': -10.0 + i, 'lon': 20.0 + i, 'time': f'{y}-06-15T00:00:00Z'})
        examples[y] = pts
    return examples


def main():
    args = parse_args()
    ensure_outdir(args.out_dir)

    # Determine available years from by_year dir
    years = []
    if os.path.isdir(args.by_year_dir):
        for fname in os.listdir(args.by_year_dir):
            if fname.startswith('combined_') and fname.endswith('.csv'):
                try:
                    y = int(fname.replace('combined_', '').replace('.csv', ''))
                    years.append(y)
                except Exception:
                    pass
    if not years:
        # fallback to a small default range
        years = list(range(2012, 2020))
    years = sorted(years)

    if args.mode == 'simulate':
        # For simulate: we will not read full missing flags. We'll build example URLs and a sampling plan.
        examples = select_example_points(args.by_year_dir, years, per_year_examples=5)

        base_wmts = 'https://wmts.marine.copernicus.eu/teroWmts'

        plan = {'layer': args.layer, 'tilematrixset': args.tilematrixset, 'tilematrix_example': args.tilematrix, 'years': {}, 'notes': 'SIMULATE mode: no network calls performed'}

        for y in years:
            pts = examples.get(y, [])
            urls = []
            for p in pts:
                lat = p['lat']
                lon = p['lon']
                # compute tile/pixel using web mercator approximation for examples
                tx, ty, px, py = latlon_to_tile_webmercator(lat, lon, args.tilematrix)
                time_str = p['time']
                url = build_getfeatureinfo_url(base_wmts, args.layer, args.tilematrixset, args.tilematrix, tx, ty, px, py, time_str)
                urls.append({'lat': lat, 'lon': lon, 'time': time_str, 'tilecol': tx, 'tilerow': ty, 'i': px, 'j': py, 'url': url})
            plan['years'][str(y)] = {'examples_count': len(urls), 'examples': urls}

        plan['sample_size_requested'] = args.sample_size
        plan['generated_at'] = datetime.utcnow().isoformat() + 'Z'

        out_path = os.path.join(args.out_dir, 'plan_simulate.json')
        with open(out_path, 'w') as f:
            json.dump(plan, f, indent=2)

        print('Simulate plan written to', out_path)
        print('Summary:')
        print('years:', ','.join(str(y) for y in years))
        print('per-year example counts:')
        for y in years:
            print(y, '->', len(plan['years'][str(y)]['examples']))
        return

    # SAMPLE mode: collect missing rows and run a throttled pilot of GetFeatureInfo requests
    if args.mode == 'sample':
        # Build list of candidate missing rows by scanning parquet outputs under data/sst_by_year
        import glob
        import time
        import threading
        import re
        import requests

        # helper to find parquet files per year
        year_parquet_files = {}
        for y in years:
            pdir = os.path.join('data', 'sst_by_year', str(y))
            if os.path.isdir(pdir):
                files = glob.glob(os.path.join(pdir, '*.parquet'))
                year_parquet_files[y] = files
            else:
                year_parquet_files[y] = []

        # determine per-year quota (even split)
        total_sample = args.sample_size
        per_year_quota = {y: max(1, total_sample // len(years)) for y in years}

        sampled_rows = []
        # read parquet files until quotas met
        for y in years:
            quota = per_year_quota.get(y, 0)
            files = year_parquet_files.get(y, [])
            got = 0
            for fpath in files:
                if got >= quota:
                    break
                try:
                    df = pd.read_parquet(fpath, columns=['latitude', 'longitude', 'time', 'sst'])
                except Exception:
                    continue
                # select rows where sst is null
                mdf = df[df['sst'].isnull()]
                if mdf.empty:
                    continue
                take = mdf.sample(min(len(mdf), quota - got))
                for _, r in take.iterrows():
                    sampled_rows.append({'year': y, 'lat': float(r['latitude']), 'lon': float(r['longitude']), 'time': pd.to_datetime(r['time']).to_pydatetime().isoformat()})
                got = sum(1 for s in sampled_rows if s['year'] == y)

        if not sampled_rows:
            print('No missing rows found in parquet outputs under data/sst_by_year — aborting sample')
            return

        # limit to requested sample_size
        sampled_rows = sampled_rows[:total_sample]

        # request parameters
        base = 'https://wmts.marine.copernicus.eu/teroWmts'
        tilematrixset = args.tilematrixset
        tilematrix = args.tilematrix
        throttle = 3.0  # requests per second
        max_workers = 4
        retries = 3

        # simple global throttle using lock and last_request_time
        last_request_time = {'t': 0.0}
        last_lock = threading.Lock()

        def throttle_wait():
            interval = 1.0 / throttle
            with last_lock:
                now = time.time()
                elapsed = now - last_request_time['t']
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                last_request_time['t'] = time.time()

        import concurrent.futures

        results = []

        def fetch_point(row):
            lat = row['lat']
            lon = row['lon']
            tstr = row['time']
            tx, ty, px, py = latlon_to_tile_webmercator(lat, lon, tilematrix)
            url = build_getfeatureinfo_url(base, args.layer, tilematrixset, tilematrix, tx, ty, px, py, tstr)
            attempt = 0
            last_err = None
            while attempt < retries:
                attempt += 1
                try:
                    throttle_wait()
                    st = time.time()
                    r = requests.get(url, timeout=30)
                    latency_ms = int((time.time() - st) * 1000)
                    status = r.status_code
                    sst_val = None
                    parsed = None
                    try:
                        parsed = r.json()
                        # try to find numeric value in JSON (search recursively)
                        def find_number(obj):
                            if isinstance(obj, dict):
                                for v in obj.values():
                                    nv = find_number(v)
                                    if nv is not None:
                                        return nv
                            elif isinstance(obj, list):
                                for v in obj:
                                    nv = find_number(v)
                                    if nv is not None:
                                        return nv
                            elif isinstance(obj, (int, float)):
                                return float(obj)
                            return None
                        sst_val = find_number(parsed)
                    except Exception:
                        # fallback: regex search in text
                        import re
                        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", r.text)
                        if m:
                            try:
                                sst_val = float(m.group(1))
                            except Exception:
                                sst_val = None
                    return {'row': row, 'url': url, 'status': status, 'latency_ms': latency_ms, 'sst_wmts': sst_val, 'attempts': attempt}
                except Exception as e:
                    last_err = str(e)
                    time.sleep(0.5 * attempt)
                    continue
            return {'row': row, 'url': url, 'status': None, 'latency_ms': None, 'sst_wmts': None, 'error': last_err, 'attempts': attempt}

        # run concurrent fetches
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fetch_point, r) for r in sampled_rows]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {'error': str(e)}
                results.append(res)
                if i % 100 == 0:
                    print(f'Completed {i}/{len(futs)} requests')

        # aggregate results per year and write parquet files
        out_reports = {}
        import math as _math
        df_all = pd.DataFrame([{
            'year': r['row']['year'], 'lat': r['row']['lat'], 'lon': r['row']['lon'], 'time': r['row']['time'],
            'sst_wmts': r.get('sst_wmts'), 'status': r.get('status'), 'latency_ms': r.get('latency_ms'), 'url': r.get('url')
        } for r in results if 'row' in r])

        if df_all.empty:
            print('No results to write from pilot')
            return

        for y, g in df_all.groupby('year'):
            pdir = os.path.join(args.out_dir)
            ensure_outdir(pdir)
            out_path = os.path.join(pdir, f'pilot_{y}.parquet')
            g.to_parquet(out_path)
            out_reports[y] = {'rows_requested': len(g), 'rows_filled': int(g['sst_wmts'].notnull().sum())}

        # write summary report
        report = {'generated_at': datetime.utcnow().isoformat() + 'Z', 'total_requested': len(df_all), 'per_year': out_reports}
        report_path = os.path.join(args.out_dir, 'report_pilot.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print('Pilot complete. Wrote per-year parquet and report at', args.out_dir)
        return

    ensure_outdir(args.out_dir)

    # Determine available years from by_year dir
    years = []
    if os.path.isdir(args.by_year_dir):
        for fname in os.listdir(args.by_year_dir):
            if fname.startswith('combined_') and fname.endswith('.csv'):
                try:
                    y = int(fname.replace('combined_', '').replace('.csv', ''))
                    years.append(y)
                except Exception:
                    pass
    if not years:
        # fallback to a small default range
        years = list(range(2012, 2020))
    years = sorted(years)

    # For simulate: we will not read full missing flags. We'll build example URLs and a sampling plan.
    examples = select_example_points(args.by_year_dir, years, per_year_examples=5)

    base_wmts = 'https://wmts.marine.copernicus.eu/teroWmts'

    plan = {'layer': args.layer, 'tilematrixset': args.tilematrixset, 'tilematrix_example': args.tilematrix, 'years': {}, 'notes': 'SIMULATE mode: no network calls performed'}

    for y in years:
        pts = examples.get(y, [])
        urls = []
        for p in pts:
            lat = p['lat']
            lon = p['lon']
            # compute tile/pixel using web mercator approximation for examples
            tx, ty, px, py = latlon_to_tile_webmercator(lat, lon, args.tilematrix)
            time_str = p['time']
            url = build_getfeatureinfo_url(base_wmts, args.layer, args.tilematrixset, args.tilematrix, tx, ty, px, py, time_str)
            urls.append({'lat': lat, 'lon': lon, 'time': time_str, 'tilecol': tx, 'tilerow': ty, 'i': px, 'j': py, 'url': url})
        plan['years'][str(y)] = {'examples_count': len(urls), 'examples': urls}

    plan['sample_size_requested'] = args.sample_size
    plan['generated_at'] = datetime.utcnow().isoformat() + 'Z'

    out_path = os.path.join(args.out_dir, 'plan_simulate.json')
    with open(out_path, 'w') as f:
        json.dump(plan, f, indent=2)

    print('Simulate plan written to', out_path)
    print('Summary:')
    print('years:', ','.join(str(y) for y in years))
    print('per-year example counts:')
    for y in years:
        print(y, '->', len(plan['years'][str(y)]['examples']))


if __name__ == '__main__':
    main()
