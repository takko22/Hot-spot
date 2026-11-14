#!/usr/bin/env python3
"""Build the Ember Atlas site into ./site with the latest NASA FIRMS fire detections.

Usage (locally or in GitHub Actions):
    FIRMS_MAP_KEY=your_key python3 scripts/fetch_firms.py

Optional settings (environment variables):
    FIRMS_SOURCES  satellites to download, comma separated (default VIIRS_NOAA20_NRT)
                   others: VIIRS_SNPP_NRT, VIIRS_NOAA21_NRT, MODIS_NRT
    FIRMS_DAYS     how many recent days to download, 1-10 (default 7)
    FIRMS_AREA     "world" or west,south,east,north  (default world)
"""
import csv, io, json, os, shutil, sys, time, urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, 'site')
API = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{area}/{days}'

def download(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            print(f'  attempt {i+1} failed: {e}')
            time.sleep(10 * (i + 1))
    return None

def main():
    # 1. copy the page and any data you keep in the repo (e.g. data/history.csv)
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, 'data'))
    shutil.copy(os.path.join(ROOT, 'index.html'), SITE)
    hist = os.path.join(ROOT, 'data', 'history.csv')
    if os.path.exists(hist):
        shutil.copy(hist, os.path.join(SITE, 'data', 'history.csv'))
        print('Included data/history.csv')
    open(os.path.join(SITE, '.nojekyll'), 'w').close()

    key = os.environ.get('FIRMS_MAP_KEY', '').strip()
    if not key:
        print('No FIRMS_MAP_KEY set: the site will run on sample data.')
        return
    sources = [s.strip() for s in os.environ.get('FIRMS_SOURCES', 'VIIRS_NOAA20_NRT').split(',') if s.strip()]
    days = max(1, min(10, int(os.environ.get('FIRMS_DAYS', '7'))))
    area = os.environ.get('FIRMS_AREA', 'world').strip() or 'world'

    # 2. download each satellite and keep only the columns the site needs
    rows = 0
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    w.writerow(['latitude', 'longitude', 'acq_date', 'frp'])
    for src in sources:
        print(f'Downloading {src}, {days} days, area {area} ...')
        txt = download(API.format(key=key, src=src, area=area, days=days))
        if not txt or not txt.lstrip().lower().startswith('latitude'):
            print('  FIRMS did not return fire data:', (txt or '').strip()[:200])
            continue
        n = 0
        for r in csv.DictReader(io.StringIO(txt)):
            try:
                lat, lon = float(r['latitude']), float(r['longitude'])
            except (KeyError, ValueError):
                continue
            frp = r.get('frp') or ''
            w.writerow([f'{lat:.4f}', f'{lon:.4f}', r.get('acq_date', ''), frp])
            n += 1
        print(f'  {n} detections')
        rows += n

    if rows == 0:
        print('No detections downloaded: the site will run on sample data.')
        return
    with open(os.path.join(SITE, 'data', 'fires.csv'), 'w') as f:
        f.write(out.getvalue())
    meta = {'updated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sources': sources, 'days': days, 'area': area, 'detections': rows}
    with open(os.path.join(SITE, 'data', 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'Wrote site/data/fires.csv with {rows} detections.')

if __name__ == '__main__':
    main()
