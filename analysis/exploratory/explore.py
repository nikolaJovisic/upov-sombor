"""Exploration of variables the original analysis never used: flow, excess sludge, biogas.
Two questions:
  A) Can biogas production be predicted from organic load and sludge? (energy recovery angle)
  B) Does modelling LOAD (conc x flow) instead of concentration reveal signal that
     concentration hides?
Same protocol throughout: chronological split, 74-day embargo, measured days only, baselines."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import numpy as np, pandas as pd, re, json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

RAW = pd.read_csv(_root / "data.csv")
hdr = RAW.iloc[0]
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
d = d.rename(columns={c: c.replace("(mg/l)_","").replace("(m3/day)_","").replace("_(m3)","")
                       .replace("_(m3/dan)","").replace("(C˚)_","") for c in d.columns})
d.columns = [re.sub(r"_+", "_", c).strip("_") for c in d.columns]
print("columns:", list(d.columns))

Q = "Q_flow"; GAS = "Vg_(gas)"; ES = "Vm_(excess_sludge)"
# plausibility filters on the operational variables the paper never cleaned
d = d[(d[Q] > 1000) & (d[Q] < 30000)]
for c in ["COD_in","BOD5_in","N_in","P_in","TSS_in"]:
    d = d[d[c].notna()]
d = d[(d["COD_in"] < 2000) & (d["BOD5_in"] < 900) & (d["N_in"] < 200) &
      (d["P_in"] < 35) & (d["TSS_in"] < 5000)]

# organic and nutrient LOADS in kg/day
for c in ["COD_in","BOD5_in","TSS_in","N_in","P_in"]:
    d[c.replace("_in","_load")] = d[c] * d[Q] / 1000.0
for c in ["COD_out","BOD5_out","TSS_out"]:
    if c in d: d[c.replace("_out","_load_out")] = d[c] * d[Q] / 1000.0

def run(target, feats, label, lags=(1,2,3,5,7,10)):
    x = d[[target, *feats]].dropna().copy()
    for L in lags:
        x[f"{target}_lag{L}"] = x[target].shift(L)
    x["roll10"] = x[target].shift(1).rolling(10, min_periods=5).mean()
    x = x.dropna()
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    if len(te) < 30 or len(tr) < 100:
        print(f"  {label:44} insufficient data (train {len(tr)}, test {len(te)})"); return
    F = [c for c in x.columns if c != target]
    sc = StandardScaler().fit(tr[F])
    Xtr, Xte = sc.transform(tr[F]), sc.transform(te[F])
    ytr, yte = tr[target], te[target]
    lr = LinearRegression().fit(Xtr, ytr)
    gb = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(Xtr, ytr)
    base = r2_score(yte, te["roll10"])
    # exogenous-only: no lag/roll of the target at all
    Fx = [c for c in feats]
    lrx = LinearRegression().fit(StandardScaler().fit_transform(tr[Fx]), ytr)
    scx = StandardScaler().fit(tr[Fx])
    r2x = r2_score(yte, lrx.predict(scx.transform(te[Fx])))
    print(f"  {label:44} n_te={len(te):4}  LR {r2_score(yte, lr.predict(Xte)):+.3f}  "
          f"GBR {r2_score(yte, gb.predict(Xte)):+.3f}  roll10 {base:+.3f}  exog-only {r2x:+.3f}")

print("\n=== A) BIOGAS PRODUCTION ===")
run(GAS, ["COD_load","BOD5_load","TSS_load",ES,Q,"Temperature_in"], "biogas ~ loads + sludge + flow + temp")
run(GAS, [ES], "biogas ~ excess sludge only")

print("\n=== B) EXCESS SLUDGE ===")
run(ES, ["COD_load","BOD5_load","TSS_load",Q], "excess sludge ~ loads + flow")

print("\n=== C) EFFLUENT AS LOAD instead of concentration ===")
for t, f in [("COD_load_out", ["COD_load","BOD5_load","TSS_load",Q,"Temperature_in","pH_in"]),
             ("BOD5_load_out", ["BOD5_load","COD_load","TSS_load",Q,"Temperature_in","pH_in"])]:
    if t in d: run(t, f, f"{t} ~ influent loads + flow")

print("\n=== D) EFFLUENT CONCENTRATION, now with flow and loads added ===")
run("COD_out", ["COD_load","BOD5_load","TSS_load",Q,"Temperature_in","pH_in","COD_in"],
    "COD_out ~ conc + loads + flow")
run("BOD5_out", ["BOD5_load","COD_load","TSS_load",Q,"Temperature_in","pH_in","BOD5_in"],
    "BOD5_out ~ conc + loads + flow")

print("\n=== E) ATTENUATION: how much variability does the plant absorb? ===")
for c in ["COD","BOD5","TSS","N","P"]:
    i, o = f"{c}_in", f"{c}_out"
    if i in d and o in d:
        s = d[[i, o]].dropna()
        cvi, cvo = s[i].std()/s[i].mean(), s[o].std()/s[o].mean()
        print(f"  {c:5} CV influent {cvi:.3f} -> CV effluent {cvo:.3f}   attenuation x{cvi/cvo:.2f}"
              f"   corr(in,out) r={s[i].corr(s[o]):+.3f}")
