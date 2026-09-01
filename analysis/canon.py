"""Canonical cleaned frame `d`, sweep-style column names. Single definition used by all figures."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd
RAW = pd.read_csv(_root / "data.csv"); hdr = RAW.iloc[0]
names, last = [], None
for c, h in zip(RAW.columns, hdr):
    if h == "Date": names.append("Date"); continue
    if "Unnamed" not in str(c): last = str(c).strip()
    names.append(re.sub(r"\s+","_", f"{last}_{str(h).strip()}".replace("nan","").strip("_ ")))
RAW = RAW.drop(index=0); RAW.columns = names
def _num(x):
    if isinstance(x,str):
        x=x.replace(",",".").strip(". ")
        try: return float(x)
        except: return np.nan
    return x
d = RAW.copy()
d = d[~d["Date"].astype(str).str.endswith("(2)")]
d["Date"] = pd.to_datetime(d["Date"].astype(str).str.replace("/",".").str.replace(",",".").str.strip("."),
                           format="%d.%m.%Y", errors="coerce")
d = d.dropna(subset=["Date"]).set_index("Date").sort_index()
for c in d.columns: d[c] = d[c].map(_num)
d.columns = [re.sub(r"_+","_",c.replace("(mg/l)_","").replace("(m3/day)_","").replace("_(m3)","")
             .replace("_(m3/dan)","").replace("(C˚)_","")).strip("_") for c in d.columns]
Q = "Q_flow"
INFL = ["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"]
OUTS = ["COD_out","BOD5_out","N_out","P_out","TSS_out"]
d = d.dropna(subset=INFL).dropna(subset=OUTS)
d = d[(d[Q] > 1000) & (d[Q] < 30000)]
d = d[~(d[INFL+OUTS] == 0).any(axis=1)]
d = d[(d.COD_out < 125) & (d.BOD5_out < 180)]
d = d[(d.COD_in < 2000) & (d.BOD5_in < 900) & (d.N_in < 200) & (d.P_in < 35) & (d.TSS_in < 5000)]
d = d.loc[:, INFL + OUTS + [Q]]          # drop gas/sludge/HRT: sparse, and not model inputs
d = d.groupby(level=0).mean().sort_index()
doy = d.index.dayofyear
d["sin_year"], d["cos_year"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
