r"""
Hotspot - GLOBAL MODIS / VIIRS active-fire harmonization (NASA Space Apps 2026)
==============================================================================

Reads the FIRMS "yearly summary by country" zips straight from a folder (no unzipping),
harmonizes MODIS (2003-2011) into VIIRS-equivalent counts, and writes map tiles for the app.

    pip install pandas numpy
    python hotspot_global.py --raw "C:\Users\tamar\Downloads" --out app\data --points-country Georgia

Quick test on one country first (minutes instead of an hour):
    python hotspot_global.py --raw "C:\Users\tamar\Downloads" --country Georgia --out app\data_georgia

Files it uses from --raw:
    modis_YYYY_all_countries.zip        2003-2025 (2000-2002 are skipped: Aqua only started in 2002)
    viirs-snpp_YYYY_all_countries.zip   2012-2025
    viirs-snpp_YYYY_*.csv               optional loose CSVs, e.g. 2026 from fetch_recent.py
Files it skips on purpose:
    viirs-jpss1_* (NOAA-20)             adding a second VIIRS satellite in 2018 would fake a jump in fires
    modis 2022-2025                     not needed: VIIRS is used from 2012, calibration ends 2021

METHOD
  1. Same filters for both sensors: vegetation fires only (type 0), MODIS confidence >= 30,
     VIIRS confidence nominal/high, VIIRS Suomi-NPP only.
  2. Common unit: fire cell-day = a ~1 km cell (0.01 deg) with >= 1 detection that day.
  3. Ratios k = VIIRS / MODIS cell-days, learned 2012-2021, per 0.5-deg block and calendar month,
     shrunk toward the 5-deg region ratio, which is shrunk toward the global ratio for that month:
         k_global,m = (V_m + a*k_all)     / (M_m + a)
         k_region,m = (V_r,m + a*k_global,m) / (M_r,m + a)
         k_block,m  = (V_b,m + a*k_region,m) / (M_b,m + a)
     A block with lots of fires gets its own ratio; a block with few borrows from its region.
  4. 2003-2011: VIIRS-equivalent = MODIS cell-days x k_block,m.  2012 onward: observed VIIRS.
  5. Validation: learn on 2012-2017, predict VIIRS 2018-2021 from MODIS, score 3 methods
     (none / one global ratio / hierarchical) at global, region and block level.

OUTPUT (in --out)
  index.json          meta, list of tiles, global monthly totals
  tiles/RI_RJ.json    one file per 5x5-degree tile: cell index i, j, month index t, count n
                      cell centre: lat = (RI*R + i + 0.5) * cell_deg,  lon = (RJ*R + j + 0.5) * cell_deg
                      (R = tile_deg / cell_deg);  month = start + t months;  t < viirs_t -> converted MODIS
  validation.json     scores + global monthly series for the pitch
  static_mask.json    ~1 km cells with known static heat sources (factories, gas flares)
  points_<country>.json   (with --points-country) every filtered detection, for map dots + CSV download
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================== SETTINGS
START_YEAR = 2003
VIIRS_START = 2012
CALIB_YEARS = (2012, 2021)    # overlap for the final ratios (before Terra/Aqua orbit drift)
FIT_YEARS = (2012, 2017)      # validation: learn here ...
TEST_YEARS = (2018, 2021)     # ... predict here
FINE = 0.01                   # fire cell-day grid (~1 km)
BLOCK = 0.5                   # ratio blocks (~50 km)
REGION = 5.0                  # ratio regions and app tiles
ALPHA = 20                    # shrinkage strength in MODIS cell-days
MODIS_MIN_CONF = 30
VIIRS_KEEP_CONF = {"n", "h"}
VIIRS_SNPP = {"N", "SUOMI NPP", "SUOMI-NPP", "SNPP", "NPP", "S-NPP"}
CHUNK_ROWS = 1_000_000        # rows read at a time from one CSV (keeps memory low)
EPOCH = np.datetime64("2000-01-01", "D")
DAY_BITS = 16384              # 2^14 days after EPOCH (to 2044) fit in the packed key
WANT = {"latitude", "longitude", "acq_date", "confidence", "type", "satellite", "frp"}
NAME_RE = re.compile(r"^(modis|viirs-snpp|viirs-jpss\d|viirs-noaa\d+)_(\d{4})", re.I)


# =================================================================== 1. FIND INPUT FILES
def discover(raw: Path, country: str | None):
    """Return {(sensor, year): [source, ...]} where a source is (kind, path, zip_member)."""
    want_suffix = f"_{country.replace(' ', '_')}.csv".lower() if country else None
    jobs, skipped = defaultdict(list), []

    def classify(name: str):
        m = NAME_RE.match(name)
        if not m:
            return None
        kind, year = m.group(1).lower(), int(m.group(2))
        sensor = {"modis": "MODIS", "viirs-snpp": "VIIRS"}.get(kind)
        if sensor is None:
            skipped.append(f"{name} (other VIIRS satellite)")
            return None
        if year < (START_YEAR if sensor == "MODIS" else VIIRS_START):
            skipped.append(f"{name} (before {START_YEAR if sensor == 'MODIS' else VIIRS_START})")
            return None
        if sensor == "MODIS" and year > CALIB_YEARS[1]:
            skipped.append(f"{name} (not needed: VIIRS covers it, calibration ends {CALIB_YEARS[1]})")
            return None
        return sensor, year

    for p in sorted(raw.glob("*.zip")):
        sy = classify(p.name)
        if not sy:
            continue
        try:
            with zipfile.ZipFile(p) as z:
                members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        except zipfile.BadZipFile:
            skipped.append(f"{p.name} (damaged zip - download it again)")
            continue
        if want_suffix:
            members = [n for n in members if n.lower().endswith(want_suffix)]
        jobs[sy] += [("zip", str(p), n) for n in members]

    for p in sorted(raw.rglob("*.csv")):
        sy = classify(p.name)
        if not sy or (want_suffix and not p.name.lower().endswith(want_suffix)):
            continue
        if any(k == "zip" for k, _, _ in jobs.get(sy, [])):
            continue  # this year is already covered by a zip; a loose copy would double count
        jobs[sy].append(("csv", str(p), None))
    return dict(jobs), skipped


def read_source(kind: str, path: str, member: str | None):
    opts = dict(usecols=lambda c: c.strip().lower() in WANT, chunksize=CHUNK_ROWS, low_memory=False)
    if kind == "zip":
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            yield from pd.read_csv(f, **opts)
    else:
        yield from pd.read_csv(path, **opts)


# =================================================================== 2. FILTER (same rules both sensors)
def fine_code(fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    return (fi + 9000) * 36001 + (fj + 18000)


def filter_chunk(df: pd.DataFrame, sensor: str, static_codes: np.ndarray | None) -> dict:
    df.columns = [c.strip().lower() for c in df.columns]
    n = len(df)
    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(float)
    fi = np.floor(lat / FINE + 1e-9)
    fj = np.floor(lon / FINE + 1e-9)
    ok = np.isfinite(fi) & np.isfinite(fj)
    fi, fj = np.where(ok, fi, 0).astype(np.int64), np.where(ok, fj, 0).astype(np.int64)
    code = fine_code(fi, fj)

    typ = pd.to_numeric(df["type"], errors="coerce").to_numpy(float) if "type" in df else np.full(n, np.nan)
    static_new = np.unique(code[(typ == 2) & ok])
    keep = (typ == 0) & ok
    untyped = np.isnan(typ) & ok                      # near-real-time files have no 'type'
    if untyped.any():
        keep |= untyped & (~np.isin(code, static_codes) if static_codes is not None else True)

    if sensor == "MODIS":
        keep &= pd.to_numeric(df["confidence"], errors="coerce").to_numpy(float) >= MODIS_MIN_CONF
    else:
        conf = df["confidence"].astype(str).str.strip().str[:1].str.lower()
        keep &= conf.isin(VIIRS_KEEP_CONF).to_numpy()
        if "satellite" in df:
            sats = df["satellite"].astype(str).str.strip().str.upper()
            if sats.nunique() > 1:
                keep &= sats.isin(VIIRS_SNPP).to_numpy()

    dt = pd.to_datetime(df["acq_date"], format="%Y-%m-%d", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(df["acq_date"], errors="coerce")
    has_day = dt.notna().to_numpy()
    day = np.zeros(n, dtype=np.int64)
    day[has_day] = (dt[has_day].to_numpy().astype("datetime64[D]") - EPOCH).astype(np.int64)
    keep &= has_day & (day >= 0) & (day < DAY_BITS)

    frp = pd.to_numeric(df["frp"], errors="coerce").to_numpy(float) if "frp" in df else np.zeros(n)
    return {"n_raw": n, "n_keep": int(keep.sum()), "static": static_new,
            "fi": fi[keep], "fj": fj[keep], "day": day[keep],
            "lat": lat[keep], "lon": lon[keep], "frp": frp[keep]}


# =================================================================== 3. ONE JOB = ONE SENSOR-YEAR
def run_job(sensor: str, year: int, sources: list, cm_path: str, static_path: str,
            cf: int, static_codes: np.ndarray | None) -> dict:
    """Read every country file of one sensor-year, reduce to fire cell-days, count per cell-month."""
    t0 = time.time()
    keys, statics, n_raw, n_keep = [], [], 0, 0
    for kind, path, member in sources:
        for chunk in read_source(kind, path, member):
            r = filter_chunk(chunk, sensor, static_codes)
            n_raw, n_keep = n_raw + r["n_raw"], n_keep + r["n_keep"]
            statics.append(r["static"])
            if len(r["day"]):
                keys.append(np.unique(fine_code(r["fi"], r["fj"]) * DAY_BITS + r["day"]))

    keys = np.unique(np.concatenate(keys)) if keys else np.array([], dtype=np.int64)
    day, cell = keys % DAY_BITS, keys // DAY_BITS
    fi, fj = cell // 36001 - 9000, cell % 36001 - 18000
    dates = pd.DatetimeIndex(EPOCH + day.astype("timedelta64[D]"))
    cm = (pd.DataFrame({"year": dates.year, "month": dates.month, "ci": fi // cf, "cj": fj // cf})
            .groupby(["year", "month", "ci", "cj"]).size().rename("n").reset_index())
    cm = cm[cm["year"] == year]  # a few files carry spill-over days from the next/previous year

    tmp = cm_path + ".tmp"
    cm.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, cm_path)
    st = np.unique(np.concatenate(statics)) if statics else np.array([], dtype=np.int64)
    pd.DataFrame({"code": st}).to_csv(static_path, index=False)
    return {"sensor": sensor, "year": year, "files": len(sources), "raw": n_raw, "kept": n_keep,
            "cell_days": int(len(keys)), "seconds": round(time.time() - t0)}


# =================================================================== 4. RATIOS + VALIDATION
def load_block_sums(cm_dir: Path, years: range, rb: int, rr: int) -> pd.DataFrame:
    parts = []
    for sensor in ("MODIS", "VIIRS"):
        for y in years:
            p = cm_dir / f"{sensor}_{y}.csv.gz"
            if not p.exists():
                continue
            cm = pd.read_csv(p)
            cm["bi"], cm["bj"], cm["ri"], cm["rj"] = cm.ci // rb, cm.cj // rb, cm.ci // rr, cm.cj // rr
            g = cm.groupby(["bi", "bj", "ri", "rj", "year", "month"])["n"].sum().reset_index()
            g["sensor"] = sensor
            parts.append(g)
    if not parts:
        raise SystemExit("No overlap-year data found - did the VIIRS and MODIS jobs run?")
    return pd.concat(parts, ignore_index=True)


def _mv(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    t = df.pivot_table(index=keys, columns="sensor", values="n", aggfunc="sum", fill_value=0)
    return t.reindex(columns=["MODIS", "VIIRS"], fill_value=0).rename(columns={"MODIS": "M", "VIIRS": "V"})


def fit_ratios(bs: pd.DataFrame, years: tuple[int, int], blocks: pd.DataFrame, a: float = ALPHA):
    """Hierarchical shrinkage: block -> region -> global, per calendar month."""
    sub = bs[bs["year"].between(*years)]
    g = _mv(sub, ["month"]).reindex(range(1, 13), fill_value=0)
    if g["M"].sum() == 0:
        raise SystemExit("No MODIS data in the calibration years.")
    k_all = g["V"].sum() / g["M"].sum()
    k_g = (g["V"] + a * k_all) / (g["M"] + a)

    r = _mv(sub, ["ri", "rj", "month"]).reset_index()
    r["k_r"] = (r["V"] + a * r["month"].map(k_g)) / (r["M"] + a)

    full = blocks[["bi", "bj", "ri", "rj"]].drop_duplicates().merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    full = full.merge(_mv(sub, ["bi", "bj", "month"]).reset_index(), on=["bi", "bj", "month"], how="left")
    full = full.merge(r[["ri", "rj", "month", "k_r"]], on=["ri", "rj", "month"], how="left")
    full = full.fillna({"M": 0, "V": 0})
    full["k_r"] = full["k_r"].fillna(full["month"].map(k_g))
    full["k"] = (full["V"] + a * full["k_r"]) / (full["M"] + a)
    full.attrs["k_all"] = float(k_all)
    full.attrs["k_global_month"] = {int(m): round(float(v), 3) for m, v in k_g.items()}
    return full[["bi", "bj", "month", "M", "V", "k_r", "k"]]


def validate(bs: pd.DataFrame, min_actual: int = 10) -> dict:
    ratios = fit_ratios(bs, FIT_YEARS, bs)
    test = bs[bs["year"].between(*TEST_YEARS)]
    mo = test[test["sensor"] == "MODIS"].merge(ratios[["bi", "bj", "month", "k"]], on=["bi", "bj", "month"], how="left")
    mo["none"] = mo["n"].astype(float)
    mo["global"] = mo["n"] * ratios.attrs["k_all"]
    mo["hierarchical"] = mo["n"] * mo["k"].fillna(ratios.attrs["k_all"])
    vi = test[test["sensor"] == "VIIRS"]
    methods = ["none", "global", "hierarchical"]

    def compare(keys):
        p = mo.groupby(keys)[methods].sum()
        act = vi.groupby(keys)["n"].sum().rename("actual")
        return p.join(act, how="outer").fillna(0)

    def score(p, a):
        ok = a >= min_actual
        ape = ((p[ok] - a[ok]).abs() / a[ok]).median() * 100 if ok.any() else np.nan
        r = np.corrcoef(p, a)[0, 1] if p.std() > 0 and a.std() > 0 else np.nan
        return {"median_abs_pct_error": None if np.isnan(ape) else round(float(ape), 1),
                "correlation": None if np.isnan(r) else round(float(r), 3),
                "total_bias_pct": round(float((p.sum() - a.sum()) / a.sum() * 100), 1),
                "units_scored": int(ok.sum())}

    out = {"fit_years": list(FIT_YEARS), "test_years": list(TEST_YEARS),
           "unit": "fire cell-days (0.01 degree cells)"}
    for level, keys in [("global", ["year", "month"]), ("region", ["ri", "rj", "year", "month"]),
                        ("block", ["bi", "bj", "year", "month"])]:
        c = compare(keys)
        out[level] = {m: score(c[m], c["actual"]) for m in methods}
        if level == "global":
            out["series"] = [{"month": f"{y}-{m:02d}", "actual": round(r.actual, 1),
                              **{k: round(r[k], 1) for k in methods}} for (y, m), r in c.iterrows()]
    return out


# =================================================================== 5. HARMONIZE + TILES
def build_tiles(cm_dir: Path, out: Path, ratios: pd.DataFrame, cf_cell: float, rb: int, rr: int) -> dict:
    years = sorted({int(p.name.split("_")[1][:4]) for p in cm_dir.glob("*.csv.gz")})
    tmp = out / "_tiles_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    k = ratios.set_index(["bi", "bj", "month"])["k"]
    glob_month = defaultdict(float)
    sources_used = {}

    for y in years:
        sensor = "MODIS" if y < VIIRS_START else "VIIRS"
        p = cm_dir / f"{sensor}_{y}.csv.gz"
        if not p.exists():
            print(f"  {y}: no {sensor} data - this year will be empty")
            continue
        cm = pd.read_csv(p)
        if sensor == "MODIS":
            idx = pd.MultiIndex.from_arrays([cm.ci // rb, cm.cj // rb, cm.month])
            cm["count"] = cm["n"] * k.reindex(idx).fillna(ratios.attrs["k_all"]).to_numpy()
        else:
            cm["count"] = cm["n"].astype(float)
        cm["t"] = (cm["year"] - START_YEAR) * 12 + cm["month"] - 1
        cm["ri"], cm["rj"] = cm.ci // rr, cm.cj // rr
        cm["i"], cm["j"] = cm.ci - cm.ri * rr, cm.cj - cm.rj * rr
        for t, v in cm.groupby("t")["count"].sum().items():
            glob_month[int(t)] += float(v)
        for (ri, rj), g in cm.groupby(["ri", "rj"]):
            g[["i", "j", "t", "count"]].to_csv(tmp / f"{ri}_{rj}.csv", mode="a", header=False,
                                               index=False, float_format="%.1f")
        sources_used[y] = sensor
        print(f"  {y}: {sensor:5s} {cm['count'].sum():12,.0f} VIIRS-equivalent fire cell-days")

    tiles_dir = out / "tiles"
    shutil.rmtree(tiles_dir, ignore_errors=True)
    tiles_dir.mkdir(parents=True)
    tiles = []
    for f in sorted(tmp.glob("*.csv")):
        ri, rj = map(int, f.stem.split("_"))
        d = pd.read_csv(f, header=None, names=["i", "j", "t", "n"])
        d = d[d["n"] > 0].sort_values(["t", "i", "j"])
        obj = {"ri": ri, "rj": rj, "i": d["i"].tolist(), "j": d["j"].tolist(),
               "t": d["t"].tolist(), "n": d["n"].round(1).tolist()}
        (tiles_dir / f"{ri}_{rj}.json").write_text(json.dumps(obj, separators=(",", ":")))
        tiles.append([ri, rj, round(float(d["n"].sum()), 1), len(d)])
    shutil.rmtree(tmp, ignore_errors=True)

    last_t = max(glob_month) if glob_month else 0
    return {"tiles": tiles, "global_monthly": [round(glob_month.get(t, 0.0), 1) for t in range(last_t + 1)],
            "end": f"{START_YEAR + last_t // 12}-{last_t % 12 + 1:02d}", "sources": sources_used}


# =================================================================== 6. POINTS FOR ONE COUNTRY (map dots + CSV)
def export_points(raw: Path, country: str, out: Path, static_codes: np.ndarray) -> None:
    jobs, _ = discover(raw, country)
    cols = defaultdict(list)
    for (sensor, year), sources in sorted(jobs.items()):
        if sensor == "MODIS" and year >= VIIRS_START:
            continue  # the calendar uses VIIRS from 2012 on, so the dots do too
        for kind, path, member in sources:
            for chunk in read_source(kind, path, member):
                r = filter_chunk(chunk, sensor, static_codes)
                cols["lat"] += np.round(r["lat"], 4).tolist()
                cols["lon"] += np.round(r["lon"], 4).tolist()
                cols["day"] += r["day"].tolist()
                cols["frp"] += np.round(np.nan_to_num(r["frp"]), 1).tolist()
                cols["sensor"] += [sensor[0]] * len(r["day"])
    path = out / f"points_{country.replace(' ', '_')}.json"
    path.write_text(json.dumps({"epoch": str(EPOCH), "country": country, **cols}, separators=(",", ":")))
    print(f"  {path.name}: {len(cols['day']):,} detections, {path.stat().st_size / 1e6:.1f} MB")


# =================================================================== MAIN
def main() -> None:
    ap = argparse.ArgumentParser(description="Global MODIS/VIIRS hotspot harmonization")
    ap.add_argument("--raw", default=r"C:\Users\tamar\Downloads", help="folder with the FIRMS zips")
    ap.add_argument("--out", default="app/data", help="where the app's data files go")
    ap.add_argument("--work", default="data/work", help="folder for intermediate files (resumable)")
    ap.add_argument("--country", default=None, help="only this country, e.g. Georgia (fast test run)")
    ap.add_argument("--cell", type=float, default=None,
                    help="app cell size in degrees: default 0.25 global, 0.1 with --country")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="zips processed in parallel (lower it if the computer runs out of memory)")
    ap.add_argument("--points-country", default=None, help="also export detection points for this country")
    ap.add_argument("--force", action="store_true", help="redo every sensor-year instead of resuming")
    args = ap.parse_args()

    cell = args.cell or (0.1 if args.country else 0.25)
    cf, rb, rr = round(cell / FINE), round(BLOCK / cell), round(REGION / cell)
    if abs(cf * FINE - cell) > 1e-9 or abs(rb * cell - BLOCK) > 1e-9 or abs(rr * cell - REGION) > 1e-9:
        raise SystemExit("--cell must divide 0.5 evenly, e.g. 0.05, 0.1, 0.125, 0.25 or 0.5")

    raw, out = Path(args.raw), Path(args.out)
    scope = args.country.replace(" ", "_") if args.country else "global"
    work = Path(args.work) / f"{scope}_cell{cell:g}"
    cm_dir, st_dir = work / "cell_months", work / "static"
    for d in (cm_dir, st_dir, out):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------ step 1-3: reduce every sensor-year
    print(f"Scanning {raw} ...")
    jobs, skipped = discover(raw, args.country)
    if not jobs:
        raise SystemExit(f"No usable FIRMS files found in {raw}")
    for s in skipped:
        print(f"  skipping {s}")
    print(f"  {len(jobs)} sensor-years to use: "
          f"MODIS {sorted(y for s, y in jobs if s == 'MODIS')}, VIIRS {sorted(y for s, y in jobs if s == 'VIIRS')}")

    def paths(s, y):
        return str(cm_dir / f"{s}_{y}.csv.gz"), str(st_dir / f"{s}_{y}.csv")

    todo = {sy: src for sy, src in jobs.items() if args.force or not Path(paths(*sy)[0]).exists()}
    zip_only = {sy: src for sy, src in todo.items() if all(k == "zip" for k, _, _ in src)}
    with_csv = {sy: src for sy, src in todo.items() if sy not in zip_only}
    print(f"  {len(jobs) - len(todo)} already done, {len(todo)} to process with {args.workers} workers\n")

    def run_batch(batch, static_codes):
        if not batch:
            return
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_job, s, y, src, *paths(s, y), cf, static_codes): (s, y)
                    for (s, y), src in sorted(batch.items(), key=lambda kv: -len(kv[1]))}
            for f in as_completed(futs):
                r = f.result()
                print(f"  done {r['sensor']:5s} {r['year']}: {r['files']:4d} files, kept {r['kept']:>11,} "
                      f"of {r['raw']:>11,} detections -> {r['cell_days']:>10,} fire cell-days ({r['seconds']}s)")

    def all_static():
        parts = [pd.read_csv(p)["code"].to_numpy() for p in st_dir.glob("*.csv") if p.stat().st_size > 5]
        return np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)

    run_batch(zip_only, None)
    run_batch(with_csv, all_static())  # loose near-real-time CSVs need the static mask from the zips
    static_codes = all_static()

    # ------------------------------------------------ step 4: ratios + validation
    print("\nLearning ratios ...")
    bs = load_block_sums(cm_dir, range(CALIB_YEARS[0], CALIB_YEARS[1] + 1), rb, rr)
    early_blocks = []
    for p in cm_dir.glob("MODIS_*.csv.gz"):
        if int(p.name.split("_")[1][:4]) < VIIRS_START:
            c = pd.read_csv(p, usecols=["ci", "cj"]).drop_duplicates()
            early_blocks.append(pd.DataFrame({"bi": c.ci // rb, "bj": c.cj // rb, "ri": c.ci // rr, "rj": c.cj // rr}))
    blocks = pd.concat([bs[["bi", "bj", "ri", "rj"]], *early_blocks]).drop_duplicates()
    ratios = fit_ratios(bs, CALIB_YEARS, blocks)
    print(f"  global VIIRS/MODIS ratio {CALIB_YEARS[0]}-{CALIB_YEARS[1]}: {ratios.attrs['k_all']:.2f}")
    print(f"  by month: {ratios.attrs['k_global_month']}")
    ratios.to_csv(work / "ratios.csv", index=False)

    print("\nValidating (fit 2012-2017, predict 2018-2021) ...")
    v = validate(bs)
    for level in ("global", "region", "block"):
        print(f"  {level} level (lower error, higher correlation = better)")
        print("    " + pd.DataFrame(v[level]).T.to_string().replace("\n", "\n    "))
    (out / "validation.json").write_text(json.dumps(v, separators=(",", ":")))

    # ------------------------------------------------ step 5: harmonize + tiles
    print("\nHarmonizing and writing tiles ...")
    res = build_tiles(cm_dir, out, ratios, cell, rb, rr)
    index = {
        "meta": {"scope": args.country or "global", "start": f"{START_YEAR}-01", "end": res["end"],
                 "viirs_t": (VIIRS_START - START_YEAR) * 12, "cell_deg": cell, "tile_deg": REGION,
                 "cells_per_tile": rr, "unit": "VIIRS-equivalent fire cell-days",
                 "calibration": f"{CALIB_YEARS[0]}-{CALIB_YEARS[1]}", "k_global": round(ratios.attrs["k_all"], 3),
                 "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "tile_format": "tiles/RI_RJ.json: i, j (cell in tile), t (months since start), n (count)"},
        "tiles": res["tiles"],
        "global_monthly": res["global_monthly"],
    }
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    fi, fj = static_codes // 36001 - 9000, static_codes % 36001 - 18000
    (out / "static_mask.json").write_text(json.dumps(
        {"cell_deg": FINE, "cells": np.column_stack([fi, fj]).tolist()}, separators=(",", ":")))

    if args.points_country:
        print(f"\nExporting detection points for {args.points_country} ...")
        export_points(raw, args.points_country, out, static_codes)

    size = sum(p.stat().st_size for p in out.rglob("*.json")) / 1e6
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. {len(res['tiles'])} tiles, "
          f"{size:.1f} MB of app data in {out}")
    print("Checks: (1) VIIRS kept >> MODIS kept per year, (2) no jump 2011 -> 2012 in the yearly totals,"
          " (3) 'hierarchical' beats 'none' clearly in validation.")


if __name__ == "__main__":
    main()
