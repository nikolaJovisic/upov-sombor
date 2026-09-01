"""Why the models say what they say. SHAP for the two models proposed for deployment.

Reviewer 1 asked for this and the request is fair. Impurity importance ranks features but
says nothing about direction, is biased toward features taking many distinct values, and
cannot explain a single day. An operator who receives an alarm on a Tuesday wants to know
what happened on Tuesday.

Two models are explained, the two the paper actually recommends:

  1. The random forest that flags effluent total nitrogen above 15 mg/L. TreeSHAP, which is
     exact for tree ensembles.
  2. The long short term memory network that forecasts effluent BOD5 ten measurements ahead.
     Gradient SHAP, the deep network estimator, since the model has two inputs and a
     recurrent path.

Writes three figures and shap_numbers.json.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]

import json
import os
import random

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from canon import d as _canon, INFL, OUTS, Q
from model import WaterNet

OUT = os.environ.get("FIGDIR", str(_root / "figures"))
os.makedirs(OUT, exist_ok=True)
BLUE, AMBER, INK, MUTED = "#3B6FB6", "#D97706", "#1f2328", "#6B7280"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "axes.edgecolor": "#c9c9c6", "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "figure.dpi": 600, "savefig.dpi": 600,
                     "savefig.bbox": "tight"})

L = 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"
LIMIT_N = 15.0
RESULTS = {}

# final.py models on influent, effluent and flow only. canon adds seasonal terms; drop them
# so the model explained here is the model reported in the paper.
D = _canon.drop(columns=["sin_year", "cos_year"])

PRETTY = {"Temperature_in": "Influent temperature", "pH_in": "Influent pH",
          "COD_in": "Influent COD", "BOD5_in": "Influent BOD5", "N_in": "Influent nitrogen",
          "P_in": "Influent phosphorus", "TSS_in": "Influent solids", "Q_flow": "Flow",
          "excroll": "Breach rate, last 10", "average": "Effluent N, 10 measurement mean"}


def label(c):
    if c in PRETTY:
        return PRETTY[c]
    if c.startswith("exc"):
        return f"Breach {c[3:]} back"
    for base, name in (("N_out_", "Effluent N"), ("N_in_", "Influent N")):
        if c.startswith(base):
            return f"{name}, {c[len(base):]} back"
    return c


# ----------------------------------------------------------------------------------
# 1. Random forest, nitrogen exceedance. TreeSHAP.
# ----------------------------------------------------------------------------------
def exceedance_frame():
    """Exactly the construction in final.py, so this explains the published model."""
    target = "N_out"
    x = D.copy()
    tin = target.replace("_out", "_in")
    for i in range(1, L + 1):
        x[f"{target}_{i}"] = x[target].shift(i)
        x[f"{tin}_{i}"] = x[tin].shift(i)
    x["average"] = x[[f"{target}_{i}" for i in range(1, L + 1)]].mean(axis=1)
    x = x.dropna()
    y = (x[target] > LIMIT_N).astype(int)
    for i in range(1, L + 1):
        x[f"exc{i}"] = y.shift(i)
    x["excroll"] = y.shift(1).rolling(L, min_periods=5).mean()
    x["_y"] = y
    x = x.dropna()
    feats = [c for c in x.columns if c not in ("_y",) + tuple(OUTS)]
    return x, feats


x, feats = exceedance_frame()
tr = x[x.index < TR_END]
te = x[(x.index >= TE_A) & (x.index <= TE_B)]
sc = StandardScaler().fit(tr[feats])
rf = RandomForestClassifier(random_state=0, n_estimators=400, min_samples_leaf=3)
rf.fit(sc.transform(tr[feats]), tr["_y"])
proba = rf.predict_proba(sc.transform(te[feats]))[:, 1]
auc = roc_auc_score(te["_y"], proba)
print(f"random forest, nitrogen above {LIMIT_N:.0f} mg/L: AUC {auc:.3f} on {len(te)} test days")
RESULTS["random_forest"] = {"auc": round(float(auc), 3), "n_test": int(len(te)),
                            "n_features": len(feats)}

names = [label(c) for c in feats]
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(sc.transform(te[feats]))
if isinstance(sv, list):          # older shap returns one array per class
    sv = sv[1]
elif sv.ndim == 3:                # newer shap returns (n, features, classes)
    sv = sv[:, :, 1]

# Colour the swarm by the real measured value, not the standardised one.
shap.summary_plot(sv, features=te[feats].values, feature_names=names,
                  max_display=14, show=False, plot_size=(6.4, 5.0))
fig = plt.gcf()
fig.suptitle("What drives a predicted nitrogen exceedance", fontsize=9.5, y=1.01)
fig.savefig(f"{OUT}/figE_shap_forest_summary.png")
plt.close(fig)
print("wrote figE_shap_forest_summary.png")

mean_abs = np.abs(sv).mean(0)
order = np.argsort(mean_abs)[::-1]
RESULTS["random_forest"]["top_features"] = [
    {"feature": names[i], "mean_abs_shap": round(float(mean_abs[i]), 4),
     "direction": "raises risk when high" if
     np.corrcoef(te[feats].values[:, i], sv[:, i])[0, 1] > 0 else "lowers risk when high"}
    for i in order[:8]]
for f in RESULTS["random_forest"]["top_features"]:
    print(f"   {f['feature']:34s} {f['mean_abs_shap']:.4f}  {f['direction']}")

# Two real days, because they say different things.
#
# A breach that follows a breach is a soft case: the model only has to carry the previous
# measurement forward. A breach that follows a compliant measurement is the alarm an
# operator would actually act on. Separating them is the point of a local explanation.
y_true = te["_y"].values
onset = np.where((y_true == 1) & (te["exc1"].values == 0))[0]
ongoing = np.where((y_true == 1) & (te["exc1"].values == 1))[0]
caught_onset = int((proba[onset] > 0.5).sum())
print(f"\n{len(onset)} of {int(y_true.sum())} breaches follow a compliant measurement. "
      f"The model calls {caught_onset} of them at a threshold of 0.5 "
      f"(highest probability {proba[onset].max():.2f}).")
RESULTS["onset"] = {"n_breaches": int(y_true.sum()), "n_onset": int(len(onset)),
                    "onset_detected_at_0.5": caught_onset,
                    "max_onset_probability": round(float(proba[onset].max()), 3),
                    "dates": [str(te.index[i].date()) for i in onset]}

expected = explainer.expected_value
expected = float(expected[1] if np.ndim(expected) > 0 else expected)


def waterfall(i, filename, heading):
    one = shap.Explanation(values=sv[i], base_values=expected,
                           data=te[feats].values[i], feature_names=names)
    shap.plots.waterfall(one, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.0, 4.4)
    fig.suptitle(heading, fontsize=9.5, y=1.02)
    fig.savefig(f"{OUT}/{filename}")
    plt.close(fig)
    print(f"wrote {filename}")
    return {"date": str(te.index[i].date()),
            "measured_N_out": round(float(te["N_out"].values[i]), 1),
            "previous_N_out": round(float(te["N_out_1"].values[i]), 1),
            "predicted_probability": round(float(proba[i]), 3),
            "base_rate": round(expected, 3),
            "contributions": [{"feature": names[j],
                               "value": round(float(te[feats].values[i, j]), 2),
                               "shap": round(float(sv[i, j]), 4)}
                              for j in np.argsort(np.abs(sv[i]))[::-1][:8]]}


alarm = int(ongoing[np.argmax(proba[ongoing])])
day = te.index[alarm]
print(f"\nalarm raised on {day.date()}: measured {te['N_out'].values[alarm]:.1f} mg/L, "
      f"model said {proba[alarm]:.2f}")
RESULTS["alarm_day"] = waterfall(
    alarm, "figF_shap_forest_day.png",
    f"Why the model flagged {day.strftime('%d %B %Y')}")

miss = int(onset[np.argmax(proba[onset])])
mday = te.index[miss]
print(f"missed onset on {mday.date()}: measured {te['N_out'].values[miss]:.1f} mg/L, "
      f"previous {te['N_out_1'].values[miss]:.1f} mg/L, model said {proba[miss]:.2f}")
RESULTS["missed_onset_day"] = waterfall(
    miss, "figH_shap_forest_missed.png",
    f"Why the model stayed quiet on {mday.strftime('%d %B %Y')}, when nitrogen breached")

# ----------------------------------------------------------------------------------
# 2. LSTM, effluent BOD5 ten measurements ahead. Gradient SHAP.
# ----------------------------------------------------------------------------------
HORIZON, TARGET = 10, "BOD5_out"


def horizon_frame(t, h):
    """The construction in horizon_final.py."""
    x = _canon.copy()
    tin = t.replace("_out", "_in")
    flat = list(INFL) + ["sin_year", "cos_year"]
    seq = []
    for i in range(0, L):
        x[f"y_{i}"] = x[t].shift(i)
        x[f"i_{i}"] = x[tin].shift(i)
        seq += [f"y_{i}", f"i_{i}"]
    x["roll"] = x[t].rolling(L, min_periods=L).mean()
    flat.append("roll")
    x["target"] = x[t].shift(-h)
    return x.dropna(subset=flat + seq + ["target"]), flat, seq


def train_lstm(tr, te, flat, seq, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scl = StandardScaler().fit(tr[flat])

    def tensors(df):
        return (torch.tensor(scl.transform(df[flat]), dtype=torch.float32),
                torch.tensor(df[seq].values.reshape(len(df), L, 2), dtype=torch.float32),
                torch.tensor(df[["target"]].values, dtype=torch.float32))

    cut = int(0.8 * len(tr))
    tra, va = tr.iloc[:cut], tr.iloc[cut + L:]
    Xtr_f, Xtr_s, ytr = tensors(tra)
    Xva_f, Xva_s, yva = tensors(va)
    Xte_f, Xte_s, _ = tensors(te)

    net = WaterNet(features_size=len(flat), seq_length=L, seq_channels=2,
                   output_size=1, mode="lstm")
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    crit = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_f, Xtr_s, ytr),
        batch_size=8, shuffle=False, drop_last=True)
    best, state = np.inf, None
    for _ in range(30):
        net.train()
        for bf, bs, by in loader:
            opt.zero_grad(); crit(net((bf, bs)), by).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = crit(net((Xva_f, Xva_s)), yva).item()
            if vl < best:
                best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return net, (Xtr_f, Xtr_s), (Xte_f, Xte_s)


hx, flat, seq = horizon_frame(TARGET, HORIZON)
htr = hx[hx.index < TR_END]
hte = hx[(hx.index >= TE_A) & (hx.index <= TE_B)]
net, (bg_f, bg_s), (te_f, te_s) = train_lstm(htr, hte, flat, seq)

from sklearn.metrics import r2_score
with torch.no_grad():
    pred = net((te_f, te_s)).numpy().ravel()
r2 = r2_score(hte["target"].values, pred)
print(f"\nLSTM, effluent BOD5 {HORIZON} measurements ahead: R2 {r2:.3f} on {len(hte)} test days")
RESULTS["lstm"] = {"r2": round(float(r2), 3), "n_test": int(len(hte)), "horizon": HORIZON}

class TwoInput(torch.nn.Module):
    """WaterNet takes its two inputs as one tuple. SHAP passes them positionally."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, flat, seq):
        return self.net((flat, seq))


rng = np.random.default_rng(0)
pick_bg = rng.choice(len(bg_f), size=min(100, len(bg_f)), replace=False)
gexp = shap.GradientExplainer(TwoInput(net), [bg_f[pick_bg], bg_s[pick_bg]])
gsv = gexp.shap_values([te_f, te_s])
flat_sv = np.asarray(gsv[0]).reshape(len(hte), len(flat))
seq_sv = np.asarray(gsv[1]).reshape(len(hte), L, 2)

# The sequence input is one variable per lag step and channel. Report it that way.
seq_names = [f"Effluent BOD5, {i} back" for i in range(L)] + \
            [f"Influent BOD5, {i} back" for i in range(L)]
seq_flat_sv = np.concatenate([seq_sv[:, :, 0], seq_sv[:, :, 1]], axis=1)
seq_values = np.concatenate([hte[[f"y_{i}" for i in range(L)]].values,
                             hte[[f"i_{i}" for i in range(L)]].values], axis=1)

all_sv = np.concatenate([flat_sv, seq_flat_sv], axis=1)
all_values = np.concatenate([hte[flat].values, seq_values], axis=1)
all_names = [PRETTY.get(c, {"roll": "Effluent BOD5, 10 measurement mean",
                            "sin_year": "Season, sine", "cos_year": "Season, cosine"}.get(c, c))
             for c in flat] + seq_names

shap.summary_plot(all_sv, features=all_values, feature_names=all_names,
                  max_display=14, show=False, plot_size=(6.4, 5.0))
fig = plt.gcf()
fig.suptitle(f"What the network uses to forecast BOD5 {HORIZON} measurements ahead",
             fontsize=9.5, y=1.01)
fig.savefig(f"{OUT}/figG_shap_lstm_summary.png")
plt.close(fig)
print("wrote figG_shap_lstm_summary.png")

lmean = np.abs(all_sv).mean(0)
lorder = np.argsort(lmean)[::-1]
RESULTS["lstm"]["top_features"] = [{"feature": all_names[i],
                                    "mean_abs_shap": round(float(lmean[i]), 4)}
                                   for i in lorder[:8]]
for f in RESULTS["lstm"]["top_features"]:
    print(f"   {f['feature']:34s} {f['mean_abs_shap']:.4f}")

json.dump(RESULTS, open(Path(__file__).with_name("shap_numbers.json"), "w"), indent=1)
print("\nwrote shap_numbers.json")
