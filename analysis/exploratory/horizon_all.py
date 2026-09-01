"""All model families at forecast horizon, so the paper can name one winner.
Same feature set, same partition, same baselines. Neural models are the original
architectures from the submitted study (WaterNet: MLP / Conv1D / LSTM)."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, random
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from model import WaterNet
from canon import d, INFL, OUTS, Q
rng = np.random.default_rng(0); L = 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"

def build(t, h):
    x = d.copy(); tin = t.replace("_out", "_in")
    flat = list(INFL) + ["sin_year", "cos_year"]; seq = []
    for i in range(0, L):
        x[f"y_{i}"] = x[t].shift(i); x[f"i_{i}"] = x[tin].shift(i); seq += [f"y_{i}", f"i_{i}"]
    x["roll"] = x[t].rolling(L, min_periods=L).mean(); flat.append("roll")
    x["persist"] = x[t]; x["target"] = x[t].shift(-h)
    return x.dropna(subset=flat + seq + ["target", "persist"]), flat, seq

def boot(y, pm, pb, n=6000):
    y, pm, pb = map(np.asarray, (y, pm, pb)); idx = rng.integers(0, len(y), (n, len(y)))
    Y, M, B = y[idx], pm[idx], pb[idx]; sst = ((Y - Y.mean(1, keepdims=True))**2).sum(1)
    dd = (((Y-B)**2).sum(1) - ((Y-M)**2).sum(1)) / sst
    return np.percentile(dd, 2.5), np.percentile(dd, 97.5)

def neural(tr, te, flat, seq, mode, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    use_seq = mode != "mlp"
    F = flat + ([] if use_seq else seq)
    sc = StandardScaler().fit(tr[F])
    def T(df):
        xs = [torch.tensor(sc.transform(df[F]), dtype=torch.float32)]
        if use_seq:
            xs.append(torch.tensor(df[seq].values.reshape(len(df), L, 2), dtype=torch.float32))
        return xs, torch.tensor(df[["target"]].values, dtype=torch.float32)
    cut = int(0.8*len(tr)); tra, va = tr.iloc[:cut], tr.iloc[cut+L:]
    (Xtr,ytr),(Xva,yva),(Xte,_) = T(tra), T(va), T(te)
    net = WaterNet(features_size=len(F), seq_length=L, seq_channels=2, output_size=1, mode=mode)
    opt = optim.Adam(net.parameters(), lr=0.005); sch = ExponentialLR(opt, gamma=0.9); crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(*Xtr, ytr), batch_size=8, shuffle=False, drop_last=True)
    f = lambda xs: net((xs[0], xs[1])) if use_seq else net(xs[0])
    best, state = np.inf, None
    for _ in range(30):
        net.train()
        for b in dl:
            *xb, yb = b; opt.zero_grad(); crit(f(xb), yb).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = crit(f(Xva), yva).item()
            if vl < best: best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return f(Xte).numpy().ravel()

for t in ["N_out", "BOD5_out"]:
    for h in (1, 5, 10):
        x, flat, seq = build(t, h)
        tr = x[x.index < TR_END]; te = x[(x.index >= TE_A) & (x.index <= TE_B)]
        y = te["target"].values
        r2p, r2r = r2_score(y, te["persist"]), r2_score(y, te["roll"])
        print(f"\n=== {t.replace('_out','')}  horizon {h}  (n={len(te)})   "
              f"persistence {r2p:+.3f}   rolling mean {r2r:+.3f}")
        F = flat + seq
        sc = StandardScaler().fit(tr[F])
        results = {}
        for nm, mdl in [("Linear regression", LinearRegression()),
                        ("Ridge (alpha=10)", Ridge(alpha=10)),
                        ("Random forest", RandomForestRegressor(random_state=0, n_estimators=300, min_samples_leaf=3)),
                        ("Gradient boosting", GradientBoostingRegressor(random_state=0, n_estimators=250, max_depth=3))]:
            results[nm] = mdl.fit(sc.transform(tr[F]), tr["target"]).predict(sc.transform(te[F]))
        for mode, nm in [("mlp","MLP"),("conv","Conv1D"),("lstm","LSTM")]:
            ps = [neural(tr, te, flat, seq, mode, s) for s in range(3)]
            results[nm] = np.mean(ps, axis=0)
            results[nm+"_sd"] = np.std([r2_score(y,p) for p in ps], ddof=1)
        for nm in ["Linear regression","Ridge (alpha=10)","Random forest","Gradient boosting","MLP","Conv1D","LSTM"]:
            p = results[nm]; r2 = r2_score(y, p)
            l1,h1 = boot(y,p,te["persist"].values); l2,h2 = boot(y,p,te["roll"].values)
            sd = results.get(nm+"_sd")
            sds = f" ±{sd:.3f}" if sd is not None else "       "
            print(f"   {nm:19}{r2:+.3f}{sds}   vs persist [{l1:+.3f},{h1:+.3f}]{'***' if l1>0 else '   '}"
                  f"   vs roll [{l2:+.3f},{h2:+.3f}]{'***' if l2>0 else ''}")
