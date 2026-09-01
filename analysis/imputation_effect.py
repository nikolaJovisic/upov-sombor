"""What gap filling does to apparent accuracy. Section 3.7 and Figure 7.

The original pipeline reindexed to daily frequency, interpolated in time, took a fifty
day rolling mean and used that to fill the gaps. Roughly half the calendar days in the
record were never sampled, so a large part of both training and evaluation consisted of
reconstructed values.

One model is trained on the gap filled series, exactly as before. It is then scored three
ways over the same test window: on the reconstructed days, on everything, and on the days
the plant actually measured. Writes imputation_numbers.json.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from dataset import load, format_date, format_strings

INFL = ["Temperature_(C˚)_in", "pH_in", "COD_(mg/l)_in", "BOD5_(mg/l)_in",
        "N_(mg/l)_in", "P_(mg/l)_in", "TSS_(mg/l)_in"]
OUTS = ["COD_(mg/l)_out", "BOD5_(mg/l)_out", "N_(mg/l)_out", "P_(mg/l)_out", "TSS_(mg/l)_out"]
QC, L = "Q_(m3/day)_flow", 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"


def measured_frame():
    d = load().dropna(subset=["Date"])
    d = d[~d["Date"].str.endswith("(2)")]
    d = d.dropna(subset=INFL).dropna(subset=OUTS)
    d = d.map(lambda x: format_date(x) if isinstance(x, str) else x)
    d["Date"] = pd.to_datetime(d["Date"], format="%d.%m.%Y")
    d = d.set_index("Date").map(format_strings).loc[:, [*INFL, *OUTS, QC]]
    d = d[(d[QC] > 1000) & (d[QC] < 30000)]
    d = d[~(d[[*INFL, *OUTS]] == 0).any(axis=1)]
    d = d[(d["COD_(mg/l)_out"] < 125) & (d["BOD5_(mg/l)_out"] < 180)]
    d = d[(d["COD_(mg/l)_in"] < 2000) & (d["BOD5_(mg/l)_in"] < 900) & (d["N_(mg/l)_in"] < 200) &
          (d["P_(mg/l)_in"] < 35) & (d["TSS_(mg/l)_in"] < 5000)]
    return d.groupby("Date").mean().sort_index()


def gap_filled(D):
    """The original reconstruction: daily reindex, time interpolation, fifty day rolling mean."""
    daily = D.asfreq("D")
    interp = daily.interpolate(method="time")
    rolling = interp.rolling(window=50, min_periods=1).mean()
    return daily.combine_first(rolling)


def lagged(x, target):
    tin = target[:-3] + "in"
    for i in range(1, L + 1):
        x[f"{target}_{i}"] = x[target].shift(i)
        x[f"{tin}_{i}"] = x[tin].shift(i)
    x["average"] = x[[f"{target}_{i}" for i in range(1, L + 1)]].mean(axis=1)
    return x.dropna()


D = measured_frame()
G = gap_filled(D)
measured_days = set(D.index)

OUT = {"n_measured_days": int(len(D)),
       "n_calendar_days": int(len(G)),
       "pct_reconstructed": round(100 * (1 - len(D) / len(G)), 1)}
print(f"measured days {len(D)} of {len(G)} calendar days "
      f"({OUT['pct_reconstructed']}% reconstructed)\n")
print(f"{'target':6}{'reconstructed':>15}{'combined':>12}{'measured':>12}")

for target in ["COD_(mg/l)_out", "BOD5_(mg/l)_out"]:
    x = lagged(G.copy(), target)
    feats = INFL + [c for c in x.columns if c.endswith(tuple(f"_{i}" for i in range(1, L + 1)))] + ["average"]
    feats = list(dict.fromkeys(feats))
    tr = x[x.index < TR_END]
    te = x[(x.index >= TE_A) & (x.index <= TE_B)]
    sc = StandardScaler().fit(tr[INFL])
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    Xtr[INFL] = sc.transform(tr[INFL]); Xte[INFL] = sc.transform(te[INFL])
    model = LinearRegression().fit(Xtr, tr[target])
    pred = pd.Series(model.predict(Xte), index=te.index)

    is_measured = te.index.isin(measured_days)
    scores = {
        "reconstructed": round(float(r2_score(te[target][~is_measured], pred[~is_measured])), 3),
        "combined": round(float(r2_score(te[target], pred)), 3),
        "measured": round(float(r2_score(te[target][is_measured], pred[is_measured])), 3),
    }
    scores["n_reconstructed"] = int((~is_measured).sum())
    scores["n_measured"] = int(is_measured.sum())
    OUT[target.split("_")[0]] = scores
    print(f"{target.split('_')[0]:6}{scores['reconstructed']:15.3f}"
          f"{scores['combined']:12.3f}{scores['measured']:12.3f}")

json.dump(OUT, open(Path(__file__).with_name("imputation_numbers.json"), "w"), indent=1)
print("\nwrote imputation_numbers.json")
