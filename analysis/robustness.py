"""Does the nitrogen advantage survive on other chronological splits?

Reported in Section 3.4. Same feature construction and same baseline as final.py,
refitted on two earlier train and test periods as well as the main one.
Writes robustness_numbers.json.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]

import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# reuse the canonical frame and constants that final.py builds, without rerunning its models
_final = (Path(__file__).with_name("final.py")).read_text().split("# ---------------- descriptive")[0]
exec(_final)

SPLITS = [("2017-01-01", "2017-04-01", "2018-01-01", "train to 2016, test 2017"),
          ("2018-01-01", "2018-04-01", "2019-01-01", "train to 2017, test 2018"),
          ("2019-02-01", "2019-04-15", "2020-03-11", "main split")]
TARGETS = ["N_(mg/l)_out", "COD_(mg/l)_out", "BOD5_(mg/l)_out"]


def frame(t):
    x = D.copy(); tin = t[:-3] + "in"; seq = []
    for i in range(1, L + 1):
        x[f"{t}_{i}"] = x[t].shift(i); x[f"{tin}_{i}"] = x[tin].shift(i)
        seq += [f"{t}_{i}", f"{tin}_{i}"]
    x["average"] = x[[f"{t}_{i}" for i in range(1, L + 1)]].mean(axis=1)
    return x.dropna(), seq


def run(t, tr_end, te_a, te_b):
    x, seq = frame(t); F = INFL + seq + ["average"]
    tr = x[x.index < tr_end]; te = x[(x.index >= te_a) & (x.index <= te_b)]
    if len(te) < 25:
        return None
    sc = StandardScaler().fit(tr[INFL])
    Xtr, Xte = tr[F].copy(), te[F].copy()
    Xtr[INFL] = sc.transform(tr[INFL]); Xte[INFL] = sc.transform(te[INFL])
    p = LinearRegression().fit(Xtr, tr[t]).predict(Xte)
    bl = te["average"].values; y = te[t].values
    idx = rng.integers(0, len(y), (4000, len(y)))
    Y, M, B = y[idx], p[idx], bl[idx]
    sst = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
    dd = (((Y - B) ** 2).sum(1) - ((Y - M) ** 2).sum(1)) / sst
    return dict(n=int(len(te)), r2=round(float(r2_score(y, p)), 3),
                base=round(float(r2_score(y, bl)), 3),
                diff=round(float(r2_score(y, p) - r2_score(y, bl)), 3),
                ci=(round(float(np.percentile(dd, 2.5)), 3),
                    round(float(np.percentile(dd, 97.5)), 3)))


OUT = {}
print(f"{'split':28}{'target':6}{'n':>5}{'R2':>8}{'base':>8}{'diff':>8}{'95% CI':>20}")
for a, b, c, lab in SPLITS:
    for t in TARGETS:
        r = run(t, a, b, c)
        if not r:
            continue
        OUT[f"{lab} | {t.split('_')[0]}"] = r
        star = "  ***" if r["ci"][0] > 0 else ""
        print(f"{lab:28}{t.split('_')[0]:6}{r['n']:5}{r['r2']:+8.3f}{r['base']:+8.3f}"
              f"{r['diff']:+8.3f}   [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]{star}")

json.dump(OUT, open(Path(__file__).with_name("robustness_numbers.json"), "w"), indent=1)
print("\nwrote robustness_numbers.json")
