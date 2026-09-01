"""Measured-days-only pipeline. No imputed targets, no imputed features.
Lags are previous *measurements* rather than previous calendar days.
Same cleaning filters, same chronological split and embargo as the original.
Scaler fit on train only."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from dataset import load, format_date, format_strings

TARGET = os.environ.get("TARGET", "COD_(mg/l)_out")
TARGET_IN = TARGET[:-3] + "in"
INFLUENT = ["Temperature_(C˚)_in","pH_in","COD_(mg/l)_in","BOD5_(mg/l)_in",
            "N_(mg/l)_in","P_(mg/l)_in","TSS_(mg/l)_in"]
L = 10

def m(y, p):
    return dict(n=int(len(y)), r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mape=float(mape(y, p)))

# ---- clean exactly as the original, then stop (no asfreq, no interpolation) ----
d = load().dropna(subset=["Date"])
d = d[~d["Date"].str.endswith("(2)")]
d = d.dropna(subset=INFLUENT).dropna(subset=[TARGET])
d = d.map(lambda x: format_date(x) if isinstance(x, str) else x)
d["Date"] = pd.to_datetime(d["Date"], format="%d.%m.%Y")
d = d.set_index("Date").map(format_strings).loc[:, [*INFLUENT, TARGET]]
d = d[~(d == 0).any(axis=1)]
d = d[d[TARGET] < (125 if TARGET.startswith("COD") else 180)]
for c, cap in [("COD_(mg/l)_in",2000),("BOD5_(mg/l)_in",900),("N_(mg/l)_in",200),
               ("P_(mg/l)_in",35),("TSS_(mg/l)_in",5000)]:
    d = d[d[c] < cap]
d = d.groupby("Date").mean().sort_index()

gaps = pd.Series(d.index).diff().dt.days.dropna()

# ---- lags over the measured sequence ----
feats = list(INFLUENT)
for i in range(1, L + 1):
    for ch in (TARGET, TARGET_IN):
        d[f"{ch}_{i}"] = d[ch].shift(i); feats.append(f"{ch}_{i}")
d["average"] = d[[f"{TARGET}_{i}" for i in range(1, L + 1)]].mean(axis=1)
feats.append("average")
d = d.dropna()

train = d[d.index < "2019-02-01"]
test = d[(d.index >= "2019-04-15") & (d.index <= "2020-03-11")]

sc = StandardScaler().fit(train[INFLUENT])                 # fit on train only
Xtr, Xte = train[feats].copy(), test[feats].copy()
Xtr[INFLUENT] = sc.transform(train[INFLUENT])
Xte[INFLUENT] = sc.transform(test[INFLUENT])
ytr, yte = train[TARGET], test[TARGET]

out = {"target": TARGET, "n_measured_days": int(len(d)),
       "median_gap_days": float(gaps.median()), "mean_gap_days": round(float(gaps.mean()), 2),
       "pct_gap_le_2days": round(100 * float((gaps <= 2).mean()), 1),
       "n_train": int(len(train)), "n_test": int(len(test)),
       "train_span": [str(train.index.min().date()), str(train.index.max().date())],
       "test_span": [str(test.index.min().date()), str(test.index.max().date())]}

lr = LinearRegression().fit(Xtr, ytr)
out["linear_regression"] = m(yte, lr.predict(Xte))
out["ridge_alpha10"] = m(yte, Ridge(alpha=10).fit(Xtr, ytr).predict(Xte))
out["baseline_10day_mean"] = m(yte, test["average"])
out["baseline_previous"] = m(yte, test[f"{TARGET}_1"])
out["baseline_train_mean"] = m(yte, np.full(len(yte), ytr.mean()))

# influent-only model: does influent chemistry predict effluent at all?
out["influent_only"] = m(yte, LinearRegression().fit(Xtr[INFLUENT], ytr).predict(Xte[INFLUENT]))

sd = Xtr.std(axis=0).replace(0, np.nan)
share = 100 * (lr.coef_ * sd).abs() / (lr.coef_ * sd).abs().sum()
lagged = [f for f in feats if TARGET in f] + ["average"]
out["importance_lagged_effluent_pct"] = round(float(share[[f for f in lagged if f in feats]].sum()), 1)
out["top_features"] = [{"f": f, "pct": round(float(share[f]), 1)}
                       for f in share.sort_values(ascending=False).index[:8]]
print("RESULT " + json.dumps(out))
