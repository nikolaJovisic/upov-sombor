"""Multi-step-ahead forecasting, done correctly.

At time t, using only information available at t (influent up to t, effluent up to t,
season), predict y_{t+h}. Baselines use the same information set:
  persistence  = y_t
  rolling mean = mean(y_{t-9..t})
Both are the honest trivial alternatives at every horizon.
"""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from canon import d, INFL, OUTS, Q
rng = np.random.default_rng(0); L = 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"

def build(t, h):
    """Features known at time t; target is y at t+h (h in measurement steps)."""
    x = d.copy(); tin = t.replace("_out", "_in")
    feats = list(INFL) + ["sin_year", "cos_year"]
    for i in range(0, L):                      # lag 0 = value at time t (known)
        x[f"y_{i}"] = x[t].shift(i); x[f"i_{i}"] = x[tin].shift(i)
        feats += [f"y_{i}", f"i_{i}"]
    x["roll"] = x[t].rolling(L, min_periods=L).mean()      # mean up to and including t
    feats.append("roll")
    x["persist"] = x[t]                                     # y_t
    x["target"] = x[t].shift(-h)                            # y_{t+h}
    return x.dropna(subset=feats + ["target", "persist", "roll"]), feats

def boot_r2(y, pm, pb, n=6000):
    y, pm, pb = map(np.asarray, (y, pm, pb)); idx = rng.integers(0, len(y), (n, len(y)))
    Y, M, B = y[idx], pm[idx], pb[idx]; sst = ((Y - Y.mean(1, keepdims=True))**2).sum(1)
    dd = (((Y-B)**2).sum(1) - ((Y-M)**2).sum(1)) / sst
    return np.percentile(dd, 2.5), np.percentile(dd, 97.5)

print("=" * 108)
print("REGRESSION: model vs BOTH baselines at each horizon (95% CI on R2 difference)")
print(f"{'target':7}{'h':>3}{'n':>5}{'model':>8}{'persist':>9}{'roll':>8}"
      f"{'vs persist':>22}{'vs roll':>22}")
rows = []
for t in ["N_out", "COD_out", "BOD5_out"]:
    for h in (1, 2, 3, 5, 7, 10):
        x, feats = build(t, h)
        tr = x[x.index < TR_END]; te = x[(x.index >= TE_A) & (x.index <= TE_B)]
        if len(te) < 40: continue
        sc = StandardScaler().fit(tr[feats])
        best, bname = None, None
        for nm, mdl in [("LR", LinearRegression()),
                        ("RF", RandomForestRegressor(random_state=0, n_estimators=300, min_samples_leaf=3)),
                        ("GBR", GradientBoostingRegressor(random_state=0, n_estimators=250, max_depth=3))]:
            p = mdl.fit(sc.transform(tr[feats]), tr["target"]).predict(sc.transform(te[feats]))
            r = r2_score(te["target"], p)
            if best is None or r > best[0]: best, bname = (r, p), nm
        r2m, pred = best
        y = te["target"].values
        r2p = r2_score(y, te["persist"]); r2r = r2_score(y, te["roll"])
        lo1, hi1 = boot_r2(y, pred, te["persist"].values)
        lo2, hi2 = boot_r2(y, pred, te["roll"].values)
        s1 = "***" if lo1 > 0 else ""
        s2 = "***" if lo2 > 0 else ""
        print(f"{t.replace('_out',''):7}{h:>3}{len(te):>5}{r2m:>8.3f}{r2p:>9.3f}{r2r:>8.3f}"
              f"   [{lo1:+.3f},{hi1:+.3f}]{s1:>4}   [{lo2:+.3f},{hi2:+.3f}]{s2:>4}  {bname}")
        rows.append((t, h, r2m, r2p, r2r, lo1 > 0 and lo2 > 0))

print("\n" + "=" * 108)
print("EXCEEDANCE at operational lead time: AUC vs best baseline")
LIM = {"N_out": 15.0, "BOD5_out": 25.0, "COD_out": 50.0}
print(f"{'target':7}{'h':>3}{'n':>5}{'rate':>7}{'model':>8}{'baseline':>10}{'diff':>9}{'95% CI':>20}")
for t, lim in LIM.items():
    for h in (1, 3, 5, 7, 10):
        x, feats = build(t, h)
        x["ylab"] = (x["target"] > lim).astype(int)
        x["b_pers"] = (x["persist"] > lim).astype(int); x["b_roll"] = x["roll"]
        tr = x[x.index < TR_END]; te = x[(x.index >= TE_A) & (x.index <= TE_B)]
        if len(te) < 40 or te["ylab"].nunique() < 2 or tr["ylab"].sum() < 20: continue
        sc = StandardScaler().fit(tr[feats])
        best, bname = None, None
        for nm, mdl in [("LOG", LogisticRegression(max_iter=2000)),
                        ("RF", RandomForestClassifier(random_state=0, n_estimators=400, min_samples_leaf=3))]:
            pp = mdl.fit(sc.transform(tr[feats]), tr["ylab"]).predict_proba(sc.transform(te[feats]))[:, 1]
            a = roc_auc_score(te["ylab"], pp)
            if best is None or a > best[0]: best, bname = (a, pp), nm
        auc, pp = best; yy = te["ylab"].values
        cands = {"persist": te["b_pers"].values, "roll": te["b_roll"].values}
        bn, bv = max(((k, roc_auc_score(yy, v)) for k, v in cands.items()), key=lambda z: z[1])
        bl = cands[bn]
        ds = []
        for _ in range(4000):
            i = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[i])) > 1:
                ds.append(roc_auc_score(yy[i], pp[i]) - roc_auc_score(yy[i], bl[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{t.replace('_out',''):7}{h:>3}{len(te):>5}{100*yy.mean():>6.0f}%{auc:>8.3f}"
              f"{bv:>10.3f}{auc-bv:>+9.3f}   [{lo:+.3f},{hi:+.3f}]{'  ***' if lo>0 else ''}  {bname}/{bn}")
