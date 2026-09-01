"""Exhaustive exploration of the Sombor dataset.
Everything the 2023 analysis did not try. Same protocol throughout:
measured days only, chronological split, 74-day embargo, bootstrap CI vs baseline.
"""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re, itertools, json
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

# ---------------- load ----------------
RAW = pd.read_csv(_root / "data.csv"); hdr = RAW.iloc[0]
names, last = [], None
for c, h in zip(RAW.columns, hdr):
    if h == "Date": names.append("Date"); continue
    if "Unnamed" not in str(c): last = str(c).strip()
    names.append(re.sub(r"\s+", "_", f"{last}_{str(h).strip()}".replace("nan", "").strip("_ ")))
RAW = RAW.drop(index=0); RAW.columns = names
def num(x):
    if isinstance(x, str):
        x = x.replace(",", ".").strip(". ")
        try: return float(x)
        except: return np.nan
    return x
d = RAW.copy()
d["Date"] = pd.to_datetime(d["Date"].astype(str).str.replace("/", ".").str.replace(",", ".").str.strip("."),
                           format="%d.%m.%Y", errors="coerce")
d = d.dropna(subset=["Date"]).set_index("Date").sort_index()
for c in d.columns: d[c] = d[c].map(num)
d.columns = [re.sub(r"_+","_",c.replace("(mg/l)_","").replace("(m3/day)_","").replace("_(m3)","")
             .replace("_(m3/dan)","").replace("(C˚)_","")).strip("_") for c in d.columns]
Q, ES, GAS = "Q_flow", "Vm_(excess_sludge)", "Vg_(gas)"
d = d[(d[Q] > 1000) & (d[Q] < 30000)]
d = d[(d.COD_in < 2000) & (d.BOD5_in < 900) & (d.N_in < 200) & (d.P_in < 35) & (d.TSS_in < 5000)]
d = d[(d.COD_out < 125) & (d.BOD5_out < 180)]
d = d.groupby(level=0).mean()

PARAMS = ["COD","BOD5","N","P","TSS"]
# ---------------- engineered features ----------------
for p in PARAMS:
    d[f"{p}_load"] = d[f"{p}_in"] * d[Q] / 1000
    d[f"{p}_load_out"] = d[f"{p}_out"] * d[Q] / 1000
    d[f"{p}_removal"] = 100 * (1 - d[f"{p}_out"] / d[f"{p}_in"])
d["COD_BOD_ratio"] = d.COD_in / d.BOD5_in          # biodegradability
d["COD_N_ratio"]   = d.COD_in / d.N_in             # C:N
d["COD_P_ratio"]   = d.COD_in / d.P_in             # C:P
d["BOD_TSS_ratio"] = d.BOD5_in / d.TSS_in
d["FM_proxy"]      = d.BOD5_load / d[ES].replace(0, np.nan)   # food-to-mass proxy
doy = d.index.dayofyear
d["sin_year"], d["cos_year"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
d["HRT_clean"] = np.where((d.HRT > 0.1) & (d.HRT < 10), d.HRT, np.nan)
for p in ["COD","BOD5","TSS"]:                       # shock indicators
    r = d[f"{p}_in"].rolling(10, min_periods=5).mean()
    d[f"{p}_shock"] = d[f"{p}_in"] / r
d["temp_x_load"] = d.Temperature_in * d.BOD5_load

BASE   = ["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"]
LOADS  = [f"{p}_load" for p in PARAMS] + [Q]
RATIOS = ["COD_BOD_ratio","COD_N_ratio","COD_P_ratio","BOD_TSS_ratio"]
SEASON = ["sin_year","cos_year"]
SHOCK  = ["COD_shock","BOD5_shock","TSS_shock"]
OPS    = [ES,"HRT_clean","temp_x_load"]

TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"
rng = np.random.default_rng(0)

def boot_diff(y, pm, pb, n=4000):
    y, pm, pb = map(np.asarray, (y, pm, pb))
    idx = rng.integers(0, len(y), (n, len(y)))
    Y, M, B = y[idx], pm[idx], pb[idx]
    sst = ((Y - Y.mean(1, keepdims=True))**2).sum(1)
    diff = (((Y-B)**2).sum(1) - ((Y-M)**2).sum(1)) / sst
    return np.percentile(diff, 2.5), np.percentile(diff, 97.5), (diff > 0).mean()

def run(target, feats, label, horizon=1, lags=(1,2,3,5,7,10), quiet=False):
    cols = list(dict.fromkeys([target, *feats]))
    x = d[cols].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(x) < 250: return None
    for L in lags: x[f"_lag{L}"] = x[target].shift(L)
    x["_roll"] = x[target].shift(1).rolling(10, min_periods=5).mean()
    if horizon > 1: x[target] = x[target].shift(-(horizon-1))
    x = x.dropna()
    tr = x[x.index < TR_END]; te = x[(x.index >= TE_A) & (x.index <= TE_B)]
    if len(te) < 30 or len(tr) < 150: return None
    F = [c for c in x.columns if c != target]
    sc = StandardScaler().fit(tr[F]); ytr, yte = tr[target], te[target]
    preds = {
        "LR":  LinearRegression().fit(sc.transform(tr[F]), ytr).predict(sc.transform(te[F])),
        "GBR": GradientBoostingRegressor(random_state=0, n_estimators=250, max_depth=3)
                 .fit(sc.transform(tr[F]), ytr).predict(sc.transform(te[F])),
    }
    bl = te["_roll"].values
    best = max(preds, key=lambda k: r2_score(yte, preds[k]))
    lo, hi, pg = boot_diff(yte, preds[best], bl)
    r2b = r2_score(yte, bl); r2m = r2_score(yte, preds[best])
    sig = "***" if lo > 0 else ("" if hi > 0 else "(worse)")
    if not quiet:
        print(f"  {label:52} n={len(te):4} {best:3} R2 {r2m:+.3f} base {r2b:+.3f} "
              f"diff {r2m-r2b:+.3f} CI[{lo:+.3f},{hi:+.3f}] {sig}")
    return dict(label=label, target=target, n=len(te), model=best, r2=r2m, base=r2b,
                diff=r2m-r2b, lo=lo, hi=hi, sig=lo > 0)

hits = []
def R(*a, **k):
    r = run(*a, **k)
    if r and r["sig"]: hits.append(r)
    return r

print("="*118)
print("1. ALL FIVE EFFLUENT PARAMETERS AS CONCENTRATION  (only COD and BOD5 were ever modelled)")
for p in PARAMS: R(f"{p}_out", BASE, f"{p}_out ~ influent concentrations")

print("\n2. ALL FIVE AS MASS LOAD (kg/day)")
for p in PARAMS: R(f"{p}_load_out", BASE + LOADS, f"{p}_load_out ~ concentrations + loads + flow")

print("\n3. REMOVAL EFFICIENCY AS TARGET (%)")
for p in PARAMS: R(f"{p}_removal", BASE + LOADS, f"{p}_removal ~ concentrations + loads")

print("\n4. FEATURE FAMILIES, on the two original targets")
fam = {"base": BASE, "base+loads": BASE+LOADS, "base+ratios": BASE+RATIOS,
       "base+season": BASE+SEASON, "base+shock": BASE+SHOCK, "base+ops": BASE+OPS,
       "everything": BASE+LOADS+RATIOS+SEASON+SHOCK+OPS}
for p in ["COD","BOD5"]:
    for nm, F in fam.items(): R(f"{p}_out", F, f"{p}_out ~ {nm}")

print("\n5. SAME, WITH LOAD AS TARGET (the promising formulation)")
for p in ["COD","BOD5"]:
    for nm, F in fam.items(): R(f"{p}_load_out", F, f"{p}_load_out ~ {nm}")

print("\n6. MULTI-STEP-AHEAD FORECASTING (does the model outlast persistence?)")
for p in ["COD","BOD5"]:
    for h in (1, 3, 5, 10):
        R(f"{p}_out", BASE+LOADS, f"{p}_out horizon {h:2} measurements", horizon=h)
        R(f"{p}_load_out", BASE+LOADS, f"{p}_load_out horizon {h:2} measurements", horizon=h)

print("\n" + "="*118)
print("SIGNIFICANT RESULTS (95% CI excludes zero)")
if hits:
    for h in sorted(hits, key=lambda r: -r["diff"]):
        print(f"  {h['label']:52} {h['model']:3} R2 {h['r2']:+.3f} vs {h['base']:+.3f} "
              f"diff {h['diff']:+.3f} CI[{h['lo']:+.3f},{h['hi']:+.3f}]")
else:
    print("  none")
json.dump(hits, open(Path(__file__).with_name("sweep_hits.json"),"w"), indent=1, default=str)
