"""Split the test set into genuinely measured days vs imputed days and score separately.
~49% of the daily series is filled, so the headline metrics mix real and synthetic targets."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape

import dataset
from dataset import (load, preprocess, format_date, format_strings,
                     FEATURE_COLUMNS, OUTPUT_COLUMN)

def m(y, p):
    return dict(n=int(len(y)), r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mape=float(mape(y, p)))

# --- reconstruct the set of dates that carry a real measurement ---
raw = load()
d = raw.dropna(subset=["Date"])
d = d[~d["Date"].str.endswith("(2)")]
d = d.dropna(subset=FEATURE_COLUMNS).dropna(subset=[OUTPUT_COLUMN])
d = d.map(lambda x: format_date(x) if isinstance(x, str) else x)
d["Date"] = pd.to_datetime(d["Date"], format="%d.%m.%Y")
d = d.set_index("Date").map(format_strings).loc[:, [*FEATURE_COLUMNS, OUTPUT_COLUMN]]
d = d[~(d == 0).any(axis=1)]
if OUTPUT_COLUMN == "COD_(mg/l)_out": d = d[d[OUTPUT_COLUMN] < 125]
if OUTPUT_COLUMN == "BOD5_(mg/l)_out": d = d[d[OUTPUT_COLUMN] < 180]
for col, cap in [("COD_(mg/l)_in",2000),("BOD5_(mg/l)_in",900),("N_(mg/l)_in",200),
                 ("P_(mg/l)_in",35),("TSS_(mg/l)_in",5000)]:
    d = d[d[col] < cap]
MEASURED = set(d.groupby("Date").mean().index)

# --- fit on the repo's own split ---
train, test = preprocess(load(), use_seq=False)
FEATS = dataset.FEATURE_COLUMNS
lr = LinearRegression().fit(train[FEATS], train[OUTPUT_COLUMN])

mask = test.index.isin(list(MEASURED))
pred = pd.Series(lr.predict(test[FEATS]), index=test.index)
y = test[OUTPUT_COLUMN]

out = {"target": OUTPUT_COLUMN,
       "test_rows": int(len(test)),
       "test_measured": int(mask.sum()),
       "test_imputed": int((~mask).sum()),
       "pct_test_imputed": round(100*(~mask).sum()/len(test), 1),
       "train_rows": int(len(train)),
       "train_measured": int(train.index.isin(list(MEASURED)).sum())}
for name, sel in [("all_test_rows", slice(None)), ("measured_only", mask), ("imputed_only", ~mask)]:
    out[name] = {
        "linear_regression": m(y[sel], pred[sel]),
        "baseline_10day_mean": m(y[sel], test["average"][sel]),
        "baseline_yesterday": m(y[sel], test[f"{OUTPUT_COLUMN}_1"][sel]),
    }
print("RESULT " + json.dumps(out))
