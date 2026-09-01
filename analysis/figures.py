"""Manuscript figures at 600 dpi. Print and grayscale safe.
Palette validated: #3B6FB6 (models) / #D97706 (baselines) — all six checks pass.
Correlation matrix uses a diverging pair with a neutral midpoint, values annotated
so colour is never the sole channel."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
from dataset import load, format_date, format_strings

OUT = os.environ.get("FIGDIR", str(_root / "figures"))
os.makedirs(OUT, exist_ok=True)
BLUE, AMBER, INK, MUTED = "#3B6FB6", "#D97706", "#1f2328", "#6B7280"
DIVERGING = LinearSegmentedColormap.from_list("amber_neutral_blue",
                                              [AMBER, "#f2f0ed", BLUE])
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "axes.edgecolor": "#c9c9c6", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "grid.color": "#e6e5e2", "grid.linewidth": 0.6,
})

INFLUENT = ["Temperature_(C˚)_in","pH_in","COD_(mg/l)_in","BOD5_(mg/l)_in",
            "N_(mg/l)_in","P_(mg/l)_in","TSS_(mg/l)_in"]
NICE = {"Temperature_(C˚)_in":"Temp$_{in}$","pH_in":"pH$_{in}$","COD_(mg/l)_in":"COD$_{in}$",
        "BOD5_(mg/l)_in":"BOD$_{5,in}$","N_(mg/l)_in":"N$_{in}$","P_(mg/l)_in":"P$_{in}$",
        "TSS_(mg/l)_in":"TSS$_{in}$"}
L = 10

def build(target):
    d = load().dropna(subset=["Date"])
    d = d[~d["Date"].str.endswith("(2)")]
    d = d.dropna(subset=INFLUENT).dropna(subset=[target])
    d = d.map(lambda x: format_date(x) if isinstance(x, str) else x)
    d["Date"] = pd.to_datetime(d["Date"], format="%d.%m.%Y")
    d = d.set_index("Date").map(format_strings).loc[:, [*INFLUENT, target]]
    d = d[~(d == 0).any(axis=1)]
    d = d[d[target] < (125 if target.startswith("COD") else 180)]
    for c, cap in [("COD_(mg/l)_in",2000),("BOD5_(mg/l)_in",900),("N_(mg/l)_in",200),
                   ("P_(mg/l)_in",35),("TSS_(mg/l)_in",5000)]:
        d = d[d[c] < cap]
    d = d.groupby("Date").mean().sort_index()
    tin = target[:-3] + "in"
    for i in range(1, L + 1):
        d[f"{target}_{i}"] = d[target].shift(i)
        d[f"{tin}_{i}"] = d[tin].shift(i)
    d["average"] = d[[f"{target}_{i}" for i in range(1, L + 1)]].mean(axis=1)
    return d.dropna()

def fit_lr(d, target):
    feats = INFLUENT + [c for c in d.columns if "_1" == c[-2:] or c == "average"
                        ] + [f"{target}_{i}" for i in range(2, L + 1)]
    feats = list(dict.fromkeys([c for c in d.columns if c != target]))
    tr = d[d.index < "2019-02-01"]; te = d[(d.index >= "2019-04-15") & (d.index <= "2020-03-11")]
    sc = StandardScaler().fit(tr[INFLUENT])
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    Xtr[INFLUENT] = sc.transform(tr[INFLUENT]); Xte[INFLUENT] = sc.transform(te[INFLUENT])
    m = LinearRegression().fit(Xtr, tr[target])
    return te, m.predict(Xte)

# ---------- Figure 2/3: correlation matrix, reduced to remove collinear lags ----------
for target, tag, lab in [("COD_(mg/l)_out","fig2","COD"),("BOD5_(mg/l)_out","fig3","BOD$_5$")]:
    d = build(target)
    tin = target[:-3] + "in"
    cols = INFLUENT + [f"{target}_1", "average", target]
    names = [NICE[c] for c in INFLUENT] + [f"{lab}$_{{out,t-1}}$", f"{lab}$_{{out}}$ 10-meas. mean",
                                           f"{lab}$_{{out}}$"]
    C = d[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            v = C.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if abs(v) > 0.55 else INK)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r", fontsize=8); cb.outline.set_visible(False)
    ax.set_title(f"Pearson correlation, {lab} predictors and effluent target\n"
                 f"measured observations only (n = {len(d)})", pad=8)
    fig.savefig(f"{OUT}/{tag}_correlation_{lab.replace('$','').replace('_','')}.png")
    plt.close(fig)

# ---------- Figure 5: measured vs predicted ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
for ax, (target, lab) in zip(axes, [("COD_(mg/l)_out","COD"),("BOD5_(mg/l)_out","BOD$_5$")]):
    d = build(target); te, pred = fit_lr(d, target)
    y = te[target].values
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, color=INK, linewidth=1.2, zorder=1, label="1:1 line")
    ax.scatter(y, pred, s=18, color=BLUE, alpha=0.75, linewidth=0.5,
               edgecolor="white", zorder=2, label="Linear regression")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"Measured effluent {lab} (mg L$^{{-1}}$)")
    ax.set_ylabel(f"Predicted effluent {lab} (mg L$^{{-1}}$)")
    ax.grid(True, linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(f"{lab}  (n = {len(y)})")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("Measured versus predicted effluent concentration, test period 15 Apr 2019 – 11 Mar 2020",
             fontsize=9, y=1.02)
fig.savefig(f"{OUT}/fig5_measured_vs_predicted.png"); plt.close(fig)

# ---------- Figure 6: model comparison against baseline ----------
# Read from final_numbers.json so the figure can never drift from Tables 2 and 3.
_nums = json.load(open(Path(__file__).with_name("final_numbers.json")))
def _bars(p):
    reg, neu = _nums["regression"][p], _nums["neural"]
    return [("LSTM", neu[f"{p}_lstm"]["r2"], 0),
            ("Linear regression", reg["lr"]["r2"], 0),
            ("MLP", neu[f"{p}_mlp"]["r2"], 1),
            ("Conv1D", neu[f"{p}_conv"]["r2"], 1),
            ("10-measurement mean", reg["roll"]["r2"], 2),
            ("Previous measurement", reg["prev"]["r2"], 2),
            ("Influent only", reg["influent_only"]["r2"], 2)]
res = {p: _bars(p) for p in ("COD", "BOD5")}

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
for ax, (tgt, lab) in zip(axes, [("COD","COD"),("BOD5","BOD$_5$")]):
    rows = res[tgt]; names = [r[0] for r in rows]; vals = [r[1] for r in rows]
    colors = [BLUE if r[2] < 2 else AMBER for r in rows]
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, vals, height=0.62, color=colors, zorder=2)
    for y_, v in zip(yy, vals):
        ax.text(v + 0.008, y_, f"{v:.3f}", va="center", fontsize=7, color=INK)
    base = next(v for n, v, _ in rows if n == "10-measurement mean")
    ax.axvline(base, color=AMBER, linestyle=(0,(4,3)), linewidth=1.2, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("R² on measured test observations"); ax.set_xlim(0, 0.52)
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.set_title(lab)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=BLUE, label="Machine learning model"),
                    Patch(facecolor=AMBER, label="Baseline / ablation")],
           frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.09))
fig.suptitle("No model exceeds the ten-measurement moving average (dashed) by a significant margin",
             fontsize=9, y=1.02)
fig.savefig(f"{OUT}/fig6_model_comparison.png"); plt.close(fig)

# ---------- Figure 7: imputed vs measured ----------
fig, ax = plt.subplots(figsize=(5.0, 3.2))
groups = ["Reconstructed\ndays only", "Combined\n(as conventionally\nreported)", "Measured\ndays only"]
_imp = json.load(open(Path(__file__).with_name("imputation_numbers.json")))
cod = [_imp["COD"][k] for k in ("reconstructed", "combined", "measured")]
bod = [_imp["BOD5"][k] for k in ("reconstructed", "combined", "measured")]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, cod, w, color=BLUE, label="COD", zorder=2)
ax.bar(x + w/2, bod, w, color=AMBER, label="BOD$_5$", zorder=2)
for xi, (a, b) in enumerate(zip(cod, bod)):
    ax.text(xi - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7, color=INK)
    ax.text(xi + w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=7.5)
ax.set_ylabel("R², linear regression"); ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Gap-filling inflates apparent performance\nidentical model, identical test period", fontsize=9)
fig.savefig(f"{OUT}/fig7_imputation_effect.png"); plt.close(fig)
print("figures written to", OUT)
