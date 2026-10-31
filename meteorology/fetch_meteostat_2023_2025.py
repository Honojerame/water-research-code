"""
Fetch meteorological daily variables from Meteostat for 2023-2025 matching the USGS data date range,
remove Daymet master, and rebuild modeling/aligned_modeling_data.csv.

Notes:
- Requires internet access and the `meteostat` package (the script will install it if missing).
- Outputs: `meteorology/meteostat_data_2023_2025.csv` and updated `modeling/aligned_modeling_data.csv`.
"""
import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime

# Install meteostat if needed
try:
    from meteostat import Point, daily, stations
except Exception:
    print('Installing meteostat...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'meteostat'])
    from meteostat import Point, daily, stations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MASTER_OUT = os.path.join(BASE_DIR, 'meteorology', 'meteostat_data_2023_2025.csv')
DAYMET_MASTER = os.path.join(BASE_DIR, 'meteorology', 'daymet_data', 'daymet_all_sites.csv')
DAYMET_BACKUP = DAYMET_MASTER + '.removed.' + datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
COORDS_PATH = os.path.join(BASE_DIR, 'USGS_Downloads', 'modeling_sites_coordinates.csv')
USGS_CLEAN = os.path.join(BASE_DIR, 'USGS_Downloads', 'modeling_data_clean.csv')
OUT_ALIGNED = os.path.join(BASE_DIR, 'modeling', 'aligned_modeling_data.csv')

# Variables to produce to match previous schema: datetime, prcp, tmax, tmin, srad, vp, swe
# Meteostat provides prcp, tavg, tmin, tmax, snow (swe proxy perhaps), tsun (sun hours)

MIN_YEAR = 2023
MAX_YEAR = 2025


def get_usgs_date_range():
    # Inspect USGS cleaned file to determine min/max dates
    sample = pd.read_csv(USGS_CLEAN, nrows=5)
    date_cols = [c for c in ['dateTime','datetime','date_time','date'] if c in sample.columns]
    parse_dates = [date_cols[0]] if date_cols else None
    usgs = pd.read_csv(USGS_CLEAN, parse_dates=parse_dates)
    dt_col = date_cols[0] if date_cols else None
    if dt_col is None:
        raise RuntimeError('No date column found in USGS cleaned data')
    usgs[dt_col] = pd.to_datetime(usgs[dt_col])
    min_date = usgs[dt_col].min().date()
    max_date = usgs[dt_col].max().date()

    # constrain to 2023-2025 window requested
    if min_date.year < MIN_YEAR:
        min_date = datetime(MIN_YEAR, 1, 1).date()
    if max_date.year > MAX_YEAR:
        max_date = datetime(MAX_YEAR, 12, 31).date()
    return min_date, max_date


def fetch_site_meteostat(site, lat, lon, start, end, pause=1.0):
    # Use Point and Daily to fetch data
    # Prefer fetching from the nearest real station(s) to improve data availability
    p = Point(lat, lon)
    s_req = stations.nearby(p)
    if hasattr(s_req, 'fetch'):
        s = s_req.fetch(5)
    else:
        s = s_req

    if s.empty:
        # no nearby stations
        return None

    # try nearest stations up to 5 until one yields data
    for st_id in s.index[:5]:
        try:
            ts = daily(st_id, start, end)
            df = ts.fetch()
        except Exception as e:
            print('    station', st_id, 'fetch error:', e)
            df = None
        if df is not None and not df.empty:
            break
    if df is None or df.empty:
        return None
    df = df.reset_index()
    # Ensure columns exist
    # Map Meteostat columns to our schema
    out = pd.DataFrame()
    out['datetime'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d') if 'time' in df.columns else pd.to_datetime(df.index).strftime('%Y-%m-%d')
    out['prcp'] = df['prcp'] if 'prcp' in df.columns else pd.NA
    out['tmax'] = df['tmax'] if 'tmax' in df.columns else pd.NA
    out['tmin'] = df['tmin'] if 'tmin' in df.columns else pd.NA
    # srad (solar radiation) not provided by Meteostat reliably; set NaN
    out['srad'] = pd.NA
    # vp (vapor pressure) not provided; set NaN
    out['vp'] = pd.NA
    # swe: Meteostat has 'snow' which is snowfall depth (mm). We'll set 'swe' to NaN (or copy snow?)
    out['swe'] = df['snow'] if 'snow' in df.columns else pd.NA
    out['site_number'] = site
    return out


def main():
    start, end = get_usgs_date_range()
    print('USGS date range to match:', start, '->', end)

    coords = pd.read_csv(COORDS_PATH, dtype={'site_number': str})
    coords['latitude'] = coords['latitude'].astype(float)
    coords['longitude'] = coords['longitude'].astype(float)

    rows = []
    for i, r in coords.iterrows():
        site = r['site_number']
        lat = r['latitude']
        lon = r['longitude']
        print(f'Fetching Meteostat for site {site} ({i+1}/{len(coords)})')
        df_site = fetch_site_meteostat(site, lat, lon, start, end)
        if df_site is None:
            print('  No data returned for site', site)
        else:
            rows.append(df_site)
            print('  rows:', len(df_site))
        time.sleep(0.8)

    if not rows:
        raise RuntimeError('No Meteostat data fetched for any site')

    master = pd.concat(rows, ignore_index=True)
    # Deduplicate
    master = master.drop_duplicates(subset=['datetime','site_number']).sort_values(['site_number','datetime'])

    # Save master
    master.to_csv(MASTER_OUT, index=False)
    print('Saved Meteostat master to', MASTER_OUT)

    # Remove Daymet master if exists (backup and delete)
    if os.path.exists(DAYMET_MASTER):
        print('Backing up and removing Daymet master...')
        os.rename(DAYMET_MASTER, DAYMET_BACKUP)
        print('Moved', DAYMET_MASTER, '->', DAYMET_BACKUP)

    # Rebuild aligned dataset the same way as notebooks/data.ipynb
    print('\nRebuilding aligned dataset using Meteostat master')
    daymet_df = master.copy()
    daymet_df['date'] = pd.to_datetime(daymet_df['datetime']).dt.date
    daymet_short = daymet_df[['site_number','date','prcp','tmax','tmin','srad','vp','swe']].copy()
    # Ensure consistent types for merging
    daymet_short['site_number'] = daymet_short['site_number'].astype(str)

    # Load USGS cleaned data
    sample = pd.read_csv(USGS_CLEAN, nrows=5)
    date_cols = [c for c in ['dateTime','datetime','date_time','date'] if c in sample.columns]
    parse_dates = [date_cols[0]] if date_cols else None
    usgs = pd.read_csv(USGS_CLEAN, parse_dates=parse_dates)
    usgs.columns = [c.strip() for c in usgs.columns]
    dt_col = None
    for c in ['dateTime','datetime','date_time','date']:
        if c in usgs.columns:
            dt_col = c
            break
    site_col = 'site_number' if 'site_number' in usgs.columns else ('site_no' if 'site_no' in usgs.columns else None)
    param_col = None
    for c in ['parameter_code','param_cd','parameter_cd']:
        if c in usgs.columns:
            param_col = c
            break
    value_col = 'value' if 'value' in usgs.columns else None
    usgs['param_norm'] = usgs[param_col].astype(str).str.lstrip('0')
    usgs_sub = usgs[usgs['param_norm'].isin(['60','65'])].copy()
    usgs_sub[dt_col] = pd.to_datetime(usgs_sub[dt_col])
    usgs_sub['date'] = usgs_sub[dt_col].dt.date

    from functools import reduce
    agg_funcs = {
        '60': {'agg_name': 'discharge_mean', 'func': 'mean'},
        '65': {'agg_name': 'gage_mean', 'func': 'mean'}
    }
    daily_list = []
    for param, meta in agg_funcs.items():
        dfp = usgs_sub[usgs_sub['param_norm'] == param].copy()
        if dfp.empty:
            continue
        grp = dfp.groupby([site_col, 'date'])[value_col].agg(meta['func']).reset_index()
        grp = grp.rename(columns={value_col: meta['agg_name']})
        daily_list.append(grp)
    if not daily_list:
        raise RuntimeError('No USGS parameters found after filtering')
    daily = reduce(lambda left, right: pd.merge(left, right, on=[site_col,'date'], how='outer'), daily_list)
    # coerce USGS site column to string to match meteostat master
    daily[site_col] = daily[site_col].astype(str)
    aligned = pd.merge(daily, daymet_short, left_on=[site_col,'date'], right_on=['site_number','date'], how='left')
    if 'site_number' in aligned.columns and site_col != 'site_number':
        aligned = aligned.drop(columns=['site_number'])
    cols_order = [site_col,'date','discharge_mean','gage_mean','prcp','tmax','tmin','srad','vp','swe']
    aligned = aligned[[c for c in cols_order if c in aligned.columns]]
    aligned.to_csv(OUT_ALIGNED, index=False)
    print('Saved aligned dataset to', OUT_ALIGNED)


if __name__ == '__main__':
    main()
