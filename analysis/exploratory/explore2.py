"""Follow-up on the two leads:
  1) biogas prediction on its own chronological split (data ends Jan 2019)
  2) BOD5 modelled as LOAD vs the moving-average baseline, with bootstrap CIs"""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

exec(open(Path(__file__).with_name("explore.py")).read().split("def run")[0])
GAS, ES, Q = "Vg_(gas)", "Vm_(excess_sludge)", "Q_flow"
rng = np.random.default_rng(0)

def boot(y, pm, pb, n=10000):
    """bootstrap 95% CI on R2(model) - R2(baseline), paired"""
    y, pm, pb = np.asarray(y), np.asarray(pm), np.asarray(pb)
    idx = rng.integers(0, len(y), (n, len(y)))
    out = []
    for i in idx:
        yi = y[i]; sst = ((yi - yi.mean())**2).sum()
        out.append((((yi-pb[i])**2).sum() - ((yi-pm[i])**2).sum())/sst)
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), (out > 0).mean()

def evaluate(x, target, feats, tr_end, te_start, te_end, label, lags=(1,2,3,5,7,10)):
    x = x[[target, *feats]].dropna().copy()
    for L in lags: x[f"lag{L}"] = x[target].shift(L)
    x["roll10"] = x[target].shift(1).rolling(10, min_periods=5).mean()
    x = x.dropna()
    tr = x[x.index < tr_end]; te = x[(x.index >= te_start) & (x.index <= te_end)]
    if len(te) < 25: print(f"  {label}: too few test points ({len(te)})"); return
    F = [c for c in x.columns if c != target]
    sc = StandardScaler().fit(tr[F])
    ytr, yte = tr[target], te[target]
    lr = LinearRegression().fit(sc.transform(tr[F]), ytr).predict(sc.transform(te[F]))
    gb = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3
                                   ).fit(sc.transform(tr[F]), ytr).predict(sc.transform(te[F]))
    bl = te["roll10"].values
    scx = StandardScaler().fit(tr[feats])
    ex = LinearRegression().fit(scx.transform(tr[feats]), ytr).predict(scx.transform(te[feats]))
    print(f"\n  {label}   train {len(tr)}  test {len(te)}  ({te.index.min().date()}..{te.index.max().date()})")
    for nm, p in [("linear regression", lr), ("gradient boosting", gb),
                  ("exogenous only (no lags)", ex)]:
        lo, hi, pg = boot(yte, p, bl)
        print(f"    {nm:26} R2 {r2_score(yte,p):+.3f}  vs baseline {r2_score(yte,bl):+.3f}  "
              f"diff {r2_score(yte,p)-r2_score(yte,bl):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p(>0)={pg:.2f}")

print("=" * 100)
print("1) BIOGAS  (Vg), own split — data ends 2019-01-31")
evaluate(d, GAS, ["COD_load","BOD5_load","TSS_load",ES,Q,"Temperature_in"],
         "2017-01-01", "2017-04-01", "2018-12-31", "biogas ~ organic load + sludge + flow + temp")
evaluate(d, GAS, [ES,"COD_load","Temperature_in"],
         "2017-01-01", "2017-04-01", "2018-12-31", "biogas ~ sludge + COD load + temp (compact)")

print("\n" + "=" * 100)
print("2) EFFLUENT AS LOAD  (kg/day) — main split")
for t, f, lab in [
    ("BOD5_load_out", ["BOD5_load","COD_load","TSS_load",Q,"Temperature_in","pH_in"], "BOD5 load out"),
    ("COD_load_out",  ["COD_load","BOD5_load","TSS_load",Q,"Temperature_in","pH_in"], "COD load out")]:
    evaluate(d, t, f, "2019-02-01", "2019-04-15", "2020-03-11", lab)

print("\n" + "=" * 100)
print("3) ATTENUATION in absolute terms (the robustness claim, done properly)")
for c in ["COD","BOD5","TSS","N","P"]:
    s = d[[f"{c}_in", f"{c}_out"]].dropna()
    si, so = s[f"{c}_in"].std(), s[f"{c}_out"].std()
    print(f"  {c:5} SD influent {si:8.1f} mg/L -> SD effluent {so:6.1f} mg/L   "
          f"absolute attenuation x{si/so:5.1f}   r(in,out) {s[f'{c}_in'].corr(s[f'{c}_out']):+.3f}   "
          f"mean removal {100*(1-s[f'{c}_out'].mean()/s[f'{c}_in'].mean()):.1f}%")
