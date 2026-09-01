"""Reconstruct dataset provenance: how many rows are measured vs imputed.
Mirrors preprocess() step by step so every number is traceable."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import os, json
import numpy as np, pandas as pd
from dataset import load, format_date, format_strings, FEATURE_COLUMNS, OUTPUT_COLUMN

steps = []
df = load(); steps.append(("rows in data.csv", len(df)))

df.dropna(subset=["Date"], inplace=True); steps.append(("after dropna(Date)", len(df)))
df = df[~df["Date"].str.endswith("(2)")]; steps.append(("after dropping '(2)' duplicates", len(df)))
df.dropna(subset=FEATURE_COLUMNS, inplace=True); steps.append(("after dropna(features)", len(df)))
df.dropna(subset=[OUTPUT_COLUMN], inplace=True); steps.append(("after dropna(target)", len(df)))

df = df.map(lambda x: format_date(x) if isinstance(x, str) else x)
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
df.set_index("Date", inplace=True)
df = df.map(format_strings)
df = df.loc[:, [*FEATURE_COLUMNS, OUTPUT_COLUMN]]

df = df[~(df == 0).any(axis=1)]; steps.append(("after dropping rows containing any 0", len(df)))
if OUTPUT_COLUMN == "COD_(mg/l)_out": df = df[df[OUTPUT_COLUMN] < 125]
if OUTPUT_COLUMN == "BOD5_(mg/l)_out": df = df[df[OUTPUT_COLUMN] < 180]
steps.append(("after effluent cap", len(df)))
for col, cap in [("COD_(mg/l)_in",2000),("BOD5_(mg/l)_in",900),("N_(mg/l)_in",200),
                 ("P_(mg/l)_in",35),("TSS_(mg/l)_in",5000)]:
    df = df[df[col] < cap]
steps.append(("after influent caps", len(df)))

df = df.groupby("Date").mean()
measured_days = len(df)
span_start, span_end = df.index.min(), df.index.max()
calendar_days = (span_end - span_start).days + 1
full = df.asfreq("D")
imputed_days = int(full[OUTPUT_COLUMN].isna().sum())

print(json.dumps({
    "target": OUTPUT_COLUMN,
    "pipeline": steps,
    "measured_days_after_cleaning": measured_days,
    "calendar_span": [str(span_start.date()), str(span_end.date())],
    "calendar_days_in_span": calendar_days,
    "days_with_no_measurement": imputed_days,
    "pct_imputed": round(100*imputed_days/calendar_days, 1),
    "pct_measured": round(100*measured_days/calendar_days, 1),
}, indent=2))
