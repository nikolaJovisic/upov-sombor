"""Exceedance prediction: can we forecast a limit breach before it happens?
This is the operationally meaningful ML task, and the ML framing of the seasonal finding.
Same protocol: measured days only, chronological split, 74-day embargo.
Baselines: base rate (climatology), persistence, and rolling mean of past exceedances."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from canon import d, INFL, OUTS, Q

rng = np.random.default_rng(0)
def boot_auc(y, p, n=4000):
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

BASE = ["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"]
LIMITS = {"N_out": 15.0, "COD_out": 50.0, "BOD5_out": 25.0, "P_out": 2.0}

def task(target, limit, feats, label, lags=(1,2,3,5,7,10)):
    x = d[list(dict.fromkeys([target, *feats]))].replace([np.inf,-np.inf], np.nan).dropna().copy()
    y = (x[target] > limit).astype(int)
    for L in lags:
        x[f"_lag{L}"] = x[target].shift(L)
        x[f"_exc{L}"] = y.shift(L)
    x["_roll"] = x[target].shift(1).rolling(10, min_periods=5).mean()
    x["_excroll"] = y.shift(1).rolling(10, min_periods=5).mean()
    x["_y"] = y
    x = x.drop(columns=[target]).dropna()
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    F = [c for c in x.columns if c != "_y"]
    ytr, yte = tr["_y"], te["_y"]
    if yte.nunique() < 2 or ytr.sum() < 20:
        print(f"  {label:46} degenerate"); return
    sc = StandardScaler().fit(tr[F])
    models = {
        "logistic":      LogisticRegression(max_iter=2000, C=1.0),
        "grad. boosting":GradientBoostingClassifier(random_state=0, n_estimators=250, max_depth=3),
        "random forest": RandomForestClassifier(random_state=0, n_estimators=400, min_samples_leaf=3),
    }
    print(f"\n  {label}   test n={len(te)}, exceedance rate {100*yte.mean():.0f}%  "
          f"(train rate {100*ytr.mean():.0f}%)")
    # baselines
    bl = {"persistence (last exceedance)": te["_exc1"].values,
          "rolling exceedance rate":       te["_excroll"].values,
          "rolling concentration":         te["_roll"].values}
    for nm, p in bl.items():
        print(f"    {'baseline: '+nm:38} AUC {roc_auc_score(yte,p):.3f}   AP {average_precision_score(yte,p):.3f}")
    best = None
    for nm, m in models.items():
        m.fit(sc.transform(tr[F]), ytr)
        p = m.predict_proba(sc.transform(te[F]))[:,1]
        auc = roc_auc_score(yte, p); ap = average_precision_score(yte, p)
        lo, hi = boot_auc(yte, p)
        print(f"    {nm:38} AUC {auc:.3f} [{lo:.3f},{hi:.3f}]   AP {ap:.3f}")
        if best is None or auc > best[1]: best = (nm, auc, p, m, F)
    # operating point at 50% threshold
    nm, auc, p, m, F = best
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    print(f"    -> best {nm}: at 0.5 threshold precision {prec:.2f}, recall {rec:.2f} "
          f"(TP {tp}, FP {fp}, FN {fn}, TN {tn})")
    if hasattr(m, "feature_importances_"):
        imp = pd.Series(m.feature_importances_, index=F).sort_values(ascending=False)
        print(f"    -> drivers: " + ", ".join(f"{k} {100*v:.0f}%" for k,v in imp.head(5).items()))

print("="*104)
print("EXCEEDANCE PREDICTION — influent chemistry + recent history")
for t, lim in LIMITS.items():
    if t in d: task(t, lim, BASE, f"{t} > {lim} mg/L")

print("\n" + "="*104)
print("DOES ADDING SEASON / TEMPERATURE HELP?  (the ML test of the summer finding)")
for t, lim in [("N_out", 15.0), ("BOD5_out", 25.0)]:
    task(t, lim, BASE + ["sin_year","cos_year"], f"{t} > {lim} mg/L  + seasonal harmonics")

print("\n" + "="*104)
print("INFLUENT + SEASON ONLY  (no effluent history at all — true forward prediction)")
for t, lim in [("N_out", 15.0), ("BOD5_out", 25.0)]:
    x = d[[t, *BASE, "sin_year", "cos_year"]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    y = (x[t] > lim).astype(int); x = x.drop(columns=[t])
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    ytr, yte = y[tr.index], y[te.index]
    sc = StandardScaler().fit(tr)
    m = GradientBoostingClassifier(random_state=0, n_estimators=250, max_depth=3).fit(sc.transform(tr), ytr)
    p = m.predict_proba(sc.transform(te))[:,1]
    lo, hi = boot_auc(yte, p)
    imp = pd.Series(m.feature_importances_, index=tr.columns).sort_values(ascending=False)
    print(f"  {t} > {lim}: AUC {roc_auc_score(yte,p):.3f} [{lo:.3f},{hi:.3f}]  "
          f"AP {average_precision_score(yte,p):.3f}  (rate {100*yte.mean():.0f}%)")
    print(f"    drivers: " + ", ".join(f"{k} {100*v:.0f}%" for k,v in imp.head(5).items()))
