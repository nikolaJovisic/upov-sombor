"""Is the nitrogen result real, and what drives it?
Checks: mechanism (temperature/nitrification), robustness across splits, ablations."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from canon import d, INFL, OUTS, Q

rng = np.random.default_rng(1)
def bootci(y, pm, pb, n=4000):
    y, pm, pb = map(np.asarray, (y, pm, pb))
    idx = rng.integers(0, len(y), (n, len(y)))
    Y, M, B = y[idx], pm[idx], pb[idx]
    sst = ((Y-Y.mean(1,keepdims=True))**2).sum(1)
    diff = (((Y-B)**2).sum(1)-((Y-M)**2).sum(1))/sst
    return np.percentile(diff,2.5), np.percentile(diff,97.5)

def fit(target, feats, tr_end, te_a, te_b, lags=(1,2,3,5,7,10), show=None):
    x = d[list(dict.fromkeys([target,*feats]))].replace([np.inf,-np.inf],np.nan).dropna().copy()
    for L in lags: x[f"_lag{L}"] = x[target].shift(L)
    x["_roll"] = x[target].shift(1).rolling(10, min_periods=5).mean()
    x = x.dropna()
    tr = x[x.index < tr_end]; te = x[(x.index >= te_a) & (x.index <= te_b)]
    if len(te) < 25: return None
    F = [c for c in x.columns if c != target]
    sc = StandardScaler().fit(tr[F])
    m = LinearRegression().fit(sc.transform(tr[F]), tr[target])
    p = m.predict(sc.transform(te[F])); bl = te["_roll"].values
    r2, r2b = r2_score(te[target], p), r2_score(te[target], bl)
    lo, hi = bootci(te[target], p, bl)
    if show:
        imp = pd.Series(np.abs(m.coef_), index=F).sort_values(ascending=False)
        print(f"    top drivers: " + ", ".join(f"{k} {100*v/imp.sum():.0f}%" for k,v in imp.head(6).items()))
    return len(te), r2, r2b, lo, hi

print("="*104)
print("A. MECHANISM — what predicts effluent nitrogen?")
sets = {
 "influent N only":            ["N_in"],
 "temperature only":           ["Temperature_in"],
 "temperature + season":       ["Temperature_in","sin_year","cos_year"],
 "influent N + temperature":   ["N_in","Temperature_in"],
 "all influent concentrations":["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"],
 "all influent, NO temperature":["pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"],
}
for nm, F in sets.items():
    r = fit("N_out", F, "2019-02-01","2019-04-15","2020-03-11", show=(nm=="all influent concentrations"))
    if r: print(f"  {nm:32} n={r[0]:4}  R2 {r[1]:+.3f}  base {r[2]:+.3f}  diff {r[1]-r[2]:+.3f}  CI[{r[3]:+.3f},{r[4]:+.3f}]")

print("\nB. Raw correlations with effluent N")
s = d[["N_out","N_in","Temperature_in","pH_in","COD_in","BOD5_in","P_in","TSS_in","Q_flow"]].dropna()
print("  " + "  ".join(f"{c} {s['N_out'].corr(s[c]):+.2f}" for c in s.columns if c != "N_out"))
_sin = pd.Series(np.sin(2*np.pi*s.index.dayofyear/365.25), index=s.index)
print(f"  N_out vs day-of-year sine: {s['N_out'].corr(_sin):+.2f}")

print("\n" + "="*104)
print("C. ROBUSTNESS — does it hold on other train/test splits?")
splits = [("2017-01-01","2017-04-01","2018-01-01","train->2016, test 2017"),
          ("2018-01-01","2018-04-01","2019-01-01","train->2017, test 2018"),
          ("2019-02-01","2019-04-15","2020-03-11","train->2019-01, test 2019-20 (main)")]
for a,b,c,lab in splits:
    for tgt in ["N_out","COD_out","BOD5_out"]:
        r = fit(tgt, ["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"], a,b,c)
        if r: print(f"  {lab:36} {tgt:9} n={r[0]:4} R2 {r[1]:+.3f} base {r[2]:+.3f} "
                    f"diff {r[1]-r[2]:+.3f} CI[{r[3]:+.3f},{r[4]:+.3f}]{'  ***' if r[3]>0 else ''}")

print("\n" + "="*104)
print("D. SEASONAL STRUCTURE of effluent nitrogen (nitrification is temperature dependent)")
m = d[["N_out","N_in","Temperature_in"]].dropna()
by = m.groupby(m.index.month).agg(N_out=("N_out","mean"), N_in=("N_in","mean"),
                                  T=("Temperature_in","mean"), n=("N_out","size"))
by["removal_%"] = 100*(1-by.N_out/by.N_in)
print(by.round(1).to_string())
print(f"\n  summer (Jun-Sep) mean effluent N: {m[m.index.month.isin([6,7,8,9])].N_out.mean():.1f} mg/L")
print(f"  winter (Dec-Mar) mean effluent N: {m[m.index.month.isin([12,1,2,3])].N_out.mean():.1f} mg/L")
print(f"  correlation effluent N vs water temperature: {m.N_out.corr(m.Temperature_in):+.3f}")
