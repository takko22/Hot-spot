"""
Hotspot - MODIS / VIIRS active-fire harmonization (NASA Space Apps 2026)
=======================================================================

One self-contained script. It turns FIRMS yearly country CSVs into one consistent
monthly fire series from 2003 to today, in "VIIRS-equivalent fire cell-days".

    pip install pandas numpy
    python hotspot_harmonize.py --raw path/to/firms_downloads --country Georgia --out app/data

Or from your own code:
    from hotspot_harmonize import run
    results = run(raw_dir="firms_downloads", country="Georgia", out_dir="app/data")

METHOD
  1. Filter both sensors the same way: vegetation fires only (type 0), drop low confidence,
     one VIIRS satellite (Suomi-NPP), MODIS Terra + Aqua from 2003.
  2. Common unit: a "fire cell-day" = a ~1 km cell (0.01 deg) with at least one detection that day.
     This removes most of the pixel-size difference (MODIS 1 km vs VIIRS 375 m).
  3. In the overlap years (2012-2021) learn k = VIIRS / MODIS for each ~50 km block and calendar month.
     Sparse blocks are pulled toward the country-wide ratio for that month (shrinkage):
         k_block,month = (V_block,month + ALPHA * k_country,month) / (M_block,month + ALPHA)
  4. 2003-2011: VIIRS-equivalent = MODIS cell-days x k.   2012 onward: observed VIIRS cell-days.
  5. Validation: learn ratios on 2012-2017, predict VIIRS for 2018-2021, compare with what VIIRS saw.

OUTPUT (in --out)
  cells.json        harmonized count per 0.1-deg cell and month   -> calendar + anomaly views
  points.json       every filtered detection (lat, lon, day, sensor, frp) -> map dots + CSV export
  validation.json   error scores for 3 methods + monthly series    -> "it works" panel / pitch slide
  static_mask.json  ~1 km cells with known static heat sources      -> cleaning live NRT data
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================== SETTINGS
START_YEAR = 2003            # first full year with Terra + Aqua
VIIRS_START = 2012           # first full year of VIIRS Suomi-NPP
CALIB_YEARS = (2012, 2021)   # overlap used for the final ratios (before Terra/Aqua orbit drift)
FIT_YEARS = (2012, 2017)     # validation: learn here ...
TEST_YEARS = (2018, 2021)    # ... predict here

FINE, CELL, BLOCK = 0.01, 0.1, 0.5   # grid sizes in degrees: cell-day unit / app cells / ratio blocks
ALPHA = 20                   # shrinkage strength, in MODIS cell-days (bigger = trust country ratio more)

MODIS_MIN_CONF = 30          # MODIS confidence 0-100
VIIRS_KEEP_CONF = {"n", "h"} # VIIRS confidence l / n / h
VIIRS_SNPP = {"N", "SUOMI NPP", "SUOMI-NPP", "SNPP", "NPP", "S-NPP"}
EPOCH = date(2000, 1, 1)     # points.json stores dates as days since this


# ============================================================== 1. LOAD + FILTER
def find_files(raw_dir: Path, country: str, sensor: str) -> list[Path]:
    """Find FIRMS yearly country files, e.g. modis_2015_Georgia.csv, viirs-snpp_2015_Georgia.csv."""
    files = sorted(raw_dir.rglob(f"*_{country.replace(' ', '_')}.csv"))
    low = lambda p: p.name.lower()
    if sensor == "MODIS":
        return [p for p in files if low(p).startswith("modis")]
    # Suomi-NPP only: skip NOAA-20 (jpss1/j1) and NOAA-21 (jpss2/j2)
    return [p for p in files if low(p).startswith("viirs")
            and not any(t in low(p) for t in ("jpss", "noaa", "j1", "j2"))]


def load_sensor(raw_dir: Path, country: str, sensor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = find_files(raw_dir, country, sensor)
    if not files:
        raise SystemExit(f"No {sensor} files for {country} under {raw_dir}")
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "type" not in df.columns:
        print(f"  WARNING: {sensor} files have no 'type' column; treating all detections as vegetation fires")
        df["type"] = 0
    n_raw = len(df)

    # static heat sources (gas flares, factories) -> mask for live data later
    static = df[df["type"] == 2]
    static_cells = pd.DataFrame({"fi": _idx(static["latitude"], FINE),
                                 "fj": _idx(static["longitude"], FINE)}).drop_duplicates()

    df = df[df["type"] == 0]
    if sensor == "MODIS":
        df = df[pd.to_numeric(df["confidence"], errors="coerce") >= MODIS_MIN_CONF]
    else:
        conf = df["confidence"].astype(str).str.strip().str[0].str.lower()
        df = df[conf.isin(VIIRS_KEEP_CONF)]
        sats = df["satellite"].astype(str).str.strip().str.upper()
        if sats.nunique() > 1:
            df = df[sats.isin(VIIRS_SNPP)]

    df = df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    d = pd.to_datetime(df["acq_date"])
    df = df.assign(acq_date=d.dt.strftime("%Y-%m-%d"), year=d.dt.year, month=d.dt.month, sensor=sensor)
    df = df[df["year"] >= (START_YEAR if sensor == "MODIS" else VIIRS_START)]
    print(f"{sensor}: {len(files)} files, kept {len(df):,} of {n_raw:,} detections")
    return df[["sensor", "latitude", "longitude", "acq_date", "acq_time", "year", "month",
               "daynight", "frp"]], static_cells


# ============================================================== 2. COMMON UNIT
def to_cell_months(det: pd.DataFrame) -> pd.DataFrame:
    """Detections -> fire cell-days (0.01 deg) -> counts per sensor, 0.1-deg cell, 0.5-deg block, month."""
    det = det.assign(fi=_idx(det["latitude"], FINE), fj=_idx(det["longitude"], FINE))
    days = det.drop_duplicates(["sensor", "fi", "fj", "acq_date"]).copy()
    lat_c, lon_c = (days["fi"] + 0.5) * FINE, (days["fj"] + 0.5) * FINE
    days["ci"], days["cj"] = _idx(lat_c, CELL), _idx(lon_c, CELL)
    days["bi"], days["bj"] = _idx(lat_c, BLOCK), _idx(lon_c, BLOCK)
    return (days.groupby(["sensor", "ci", "cj", "bi", "bj", "year", "month"])
                .size().rename("n").reset_index())


# ============================================================== 3. RATIOS
def fit_ratios(cm: pd.DataFrame, years: tuple[int, int], alpha: float = ALPHA) -> pd.DataFrame:
    """k per (block, month) = (V + alpha * k_country_month) / (M + alpha), summed over `years`."""
    sub = cm[cm["year"].between(*years)]

    def sums(keys):
        t = sub.pivot_table(index=keys, columns="sensor", values="n", aggfunc="sum", fill_value=0)
        return t.reindex(columns=["MODIS", "VIIRS"], fill_value=0).rename(columns={"MODIS": "M", "VIIRS": "V"})

    month = sums(["month"]).reindex(range(1, 13), fill_value=0)
    if month["M"].sum() == 0:
        raise SystemExit("No MODIS data in the calibration years - check the input files.")
    k_all = month["V"].sum() / month["M"].sum()
    # country-month ratio, itself shrunk toward the all-year ratio
    k_month = (month["V"] + alpha * k_all) / (month["M"] + alpha)

    blocks = cm[["bi", "bj"]].drop_duplicates().merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    r = blocks.merge(sums(["bi", "bj", "month"]).reset_index(), on=["bi", "bj", "month"], how="left").fillna(0)
    km = r["month"].map(k_month)
    r["k"] = (r["V"] + alpha * km) / (r["M"] + alpha)
    r.attrs["k_all"] = float(k_all)
    return r


def apply_ratios(modis: pd.DataFrame, ratios: pd.DataFrame) -> np.ndarray:
    m = modis[["bi", "bj", "month", "n"]].merge(ratios[["bi", "bj", "month", "k"]],
                                                on=["bi", "bj", "month"], how="left")
    return (m["n"] * m["k"].fillna(ratios.attrs["k_all"])).to_numpy()


def harmonize(cm: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    early = cm[(cm["sensor"] == "MODIS") & (cm["year"] < VIIRS_START)].copy()
    early["count"], early["source"] = apply_ratios(early, ratios), "M"
    late = cm[(cm["sensor"] == "VIIRS") & (cm["year"] >= VIIRS_START)].copy()
    late["count"], late["source"] = late["n"].astype(float), "V"
    return pd.concat([early, late], ignore_index=True)[["ci", "cj", "year", "month", "count", "source"]]


# ============================================================== 4. VALIDATION
def validate(cm: pd.DataFrame, min_actual: int = 10) -> dict:
    ratios = fit_ratios(cm, FIT_YEARS)
    test = cm[cm["year"].between(*TEST_YEARS)]
    modis = test[test["sensor"] == "MODIS"].copy()
    modis["none"] = modis["n"].astype(float)
    modis["global"] = modis["n"] * ratios.attrs["k_all"]
    modis["block_month"] = apply_ratios(modis, ratios)
    methods = ["none", "global", "block_month"]

    def compare(keys):
        pred = modis.groupby(keys)[methods].sum()
        act = test[test["sensor"] == "VIIRS"].groupby(keys)["n"].sum().rename("actual")
        return pred.join(act, how="outer").fillna(0)

    def score(p, a):
        ok = a >= min_actual
        ape = ((p[ok] - a[ok]).abs() / a[ok]).median() * 100
        r = np.corrcoef(p, a)[0, 1] if p.std() > 0 and a.std() > 0 else np.nan
        return {"median_abs_pct_error": round(float(ape), 1), "correlation": round(float(r), 3),
                "total_bias_pct": round(float((p.sum() - a.sum()) / a.sum() * 100), 1),
                "months_scored": int(ok.sum())}

    country = compare(["year", "month"])
    blocks = compare(["bi", "bj", "year", "month"])
    return {
        "fit_years": list(FIT_YEARS), "test_years": list(TEST_YEARS),
        "unit": "fire cell-days (0.01 degree cells)",
        "country": {m: score(country[m], country["actual"]) for m in methods},
        "blocks": {m: score(blocks[m], blocks["actual"]) for m in methods},
        "series": [{"month": f"{y}-{m:02d}", "actual": round(r.actual, 1), "none": round(r.none, 1),
                    "global": round(r["global"], 1), "block_month": round(r.block_month, 1)}
                   for (y, m), r in country.iterrows()],
    }


# ============================================================== 5. EXPORT
def export(out_dir: Path, country: str, h: pd.DataFrame, det: pd.DataFrame,
           static_cells: pd.DataFrame, validation: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = h[h["count"] > 0]
    ym = h["year"].astype(str) + "-" + h["month"].map("{:02d}".format)
    cells = {
        "meta": {"country": country, "start": f"{h['year'].min()}-01", "end": max(ym), "cell_deg": CELL,
                 "unit": "VIIRS-equivalent fire cell-days", "calibration": f"{CALIB_YEARS[0]}-{CALIB_YEARS[1]}",
                 "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "row_format": ["lat", "lon", "month", "count", "source (M=converted MODIS, V=VIIRS)"]},
        "rows": [[round((ci + 0.5) * CELL, 3), round((cj + 0.5) * CELL, 3), m, round(float(c), 1), s]
                 for ci, cj, m, c, s in zip(h["ci"], h["cj"], ym, h["count"], h["source"])],
    }
    days = (pd.to_datetime(det["acq_date"]) - pd.Timestamp(EPOCH)).dt.days
    points = {"epoch": EPOCH.isoformat(), "lat": det["latitude"].round(4).tolist(),
              "lon": det["longitude"].round(4).tolist(), "day": days.tolist(),
              "sensor": det["sensor"].str[0].tolist(), "frp": det["frp"].round(1).fillna(0).tolist()}
    mask = {"cell_deg": FINE, "cells": static_cells[["fi", "fj"]].values.tolist()}

    for name, obj in [("cells.json", cells), ("points.json", points),
                      ("validation.json", validation), ("static_mask.json", mask)]:
        (out_dir / name).write_text(json.dumps(obj, separators=(",", ":")))
        print(f"  wrote {name:17s} {(out_dir / name).stat().st_size / 1e6:6.2f} MB")


# ============================================================== RUN EVERYTHING
def run(raw_dir: str | Path, country: str = "Georgia", out_dir: str | Path = "app/data") -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)

    print("1. Load + filter")
    modis, s1 = load_sensor(raw_dir, country, "MODIS")
    viirs, s2 = load_sensor(raw_dir, country, "VIIRS")
    det = pd.concat([modis, viirs], ignore_index=True)
    static_cells = pd.concat([s1, s2]).drop_duplicates()
    print(det.pivot_table(index="year", columns="sensor", values="latitude",
                          aggfunc="count", fill_value=0).to_string())

    print("\n2. Fire cell-days on a ~1 km grid")
    cm = to_cell_months(det)

    print(f"\n3. Ratios from {CALIB_YEARS[0]}-{CALIB_YEARS[1]}")
    ratios = fit_ratios(cm, CALIB_YEARS)
    print(f"  country-wide VIIRS/MODIS ratio: {ratios.attrs['k_all']:.2f}")
    h = harmonize(cm, ratios)
    yearly = h.groupby("year")["count"].sum().round(0)
    print("  harmonized total per year (check: no jump between 2011 and 2012)")
    print(yearly.to_string())

    print(f"\n4. Validation: fit {FIT_YEARS}, test {TEST_YEARS}")
    v = validate(cm)
    for level in ("country", "blocks"):
        print(f"  {level} monthly totals")
        print(pd.DataFrame(v[level]).T.to_string())

    print("\n5. Export")
    export(out_dir, country, h, det, static_cells, v)
    return {"detections": det, "cell_months": cm, "ratios": ratios,
            "harmonized": h, "validation": v}


def _idx(values: pd.Series, size: float) -> pd.Series:
    return np.floor(values.astype(float) / size + 1e-9).astype(int)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harmonize FIRMS MODIS + VIIRS hotspots")
    ap.add_argument("--raw", required=True, help="folder with FIRMS yearly country CSVs (searched recursively)")
    ap.add_argument("--country", default="Georgia", help="country name as in the file names")
    ap.add_argument("--out", default="app/data", help="where to write the JSON files")
    args = ap.parse_args()
    run(args.raw, args.country, args.out)
