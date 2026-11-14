# Ember Atlas

A 3D globe and map of every fire NASA satellites detect. Search any place to see whether it is burning, its fire history and the trend since 2000.

This repository publishes the site on GitHub Pages and refreshes the fire data from NASA FIRMS every 3 hours.

## Files

| File | What it does |
|---|---|
| `index.html` | The whole website. It loads `data/fires.csv` (and `data/history.csv` if present) when it opens. |
| `scripts/fetch_firms.py` | Downloads the latest detections from the FIRMS API and builds the site into `site/`. |
| `.github/workflows/update-fires.yml` | Runs the script every 3 hours and publishes `site/` to GitHub Pages. |
| `data/history.csv` | Optional. Older detections for the "trend since 2000" charts (see below). |

## Put it online (one time)

1. **Get a free FIRMS key:** https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. **Create a GitHub repository** and upload all these files, keeping the folders (`.github/workflows/…`, `scripts/…`).
   Tip: the `.github` folder is hidden on Mac/Linux; drag the whole folder into GitHub's "upload files" page, or use `git`.
3. **Add the key as a secret:** repository → Settings → Secrets and variables → Actions → New repository secret.
   Name: `FIRMS_MAP_KEY` · Value: your key.
4. **Turn on Pages:** Settings → Pages → Build and deployment → Source: **GitHub Actions**.
5. **Run it:** Actions tab → "Update fires and publish site" → **Run workflow**.
   After about a minute the site is live at `https://<your-username>.github.io/<repo-name>/`.

From then on it updates by itself every 3 hours, and every time you push a change.

## Run it on your own computer

```bash
FIRMS_MAP_KEY=your_key python3 scripts/fetch_firms.py   # Windows PowerShell: $env:FIRMS_MAP_KEY="your_key"; python scripts/fetch_firms.py
cd site
python3 -m http.server 8000
```
Open http://localhost:8000. (Opening `index.html` by double-click also works, but then it shows sample data, because browsers block reading the CSV from a `file://` page.)

## Settings

Change these in `.github/workflows/update-fires.yml`:

- `FIRMS_SOURCES` – satellites, comma separated: `VIIRS_NOAA20_NRT` (default), `VIIRS_SNPP_NRT`, `VIIRS_NOAA21_NRT`, `MODIS_NRT`. Several satellites often see the same fire, so counts go up when you add more.
- `FIRMS_DAYS` – recent days to download, 1–10 (default 7, which fills the "this week" numbers).
- `FIRMS_AREA` – `world`, or `west,south,east,north` for one region (e.g. Australia: `112,-44,154,-10`).

## Fire history since 2000 (optional)

The live feed only covers the last few days. For the long-term trend charts, request an archive from FIRMS (**Archive Download** on the FIRMS site, e.g. MODIS for one country from 2000), then save it as `data/history.csv` in this repository.
Only `latitude`, `longitude`, `acq_date` and `frp` are needed; deleting the other columns keeps the file small (GitHub refuses files over 100 MB).
Make the archive end before the live window starts, so no day is counted twice.

## Without a key

Everything still works on built-in sample data, and anyone can load a FIRMS CSV by hand with the **Data** button in the top bar.
