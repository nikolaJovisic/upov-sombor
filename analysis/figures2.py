"""New manuscript figures for the reframed paper. 600 dpi, print and greyscale safe.
Palette validated: #3B6FB6 / #D97706 (all six checks pass)."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from canon import d, INFL, OUTS, Q

OUT = os.environ.get("FIGDIR", str(_root / "figures")); os.makedirs(OUT, exist_ok=True)
BLUE, AMBER, INK, MUTED = "#3B6FB6", "#D97706", "#1f2328", "#6B7280"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "axes.edgecolor": "#c9c9c6", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "grid.color": "#e6e5e2", "grid.linewidth": 0.6})
def clean(ax, which=("top","right")):
    for s in which: ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

BASE = ["Temperature_in","pH_in","COD_in","BOD5_in","N_in","P_in","TSS_in"]
SUM = [6,7,8]

# ================= FIG A: seasonal degradation =================
m = d[["N_out","N_in","BOD5_out","COD_out","P_out","P_in","Temperature_in"]].dropna()
m["N_rem"] = 100*(1-m.N_out/m.N_in); m["P_rem"] = 100*(1-m.P_out/m.P_in)
by = m.groupby(m.index.month).agg(N_rem=("N_rem","mean"), P_rem=("P_rem","mean"),
                                  N_out=("N_out","mean"), BOD=("BOD5_out","mean"),
                                  T=("Temperature_in","mean"))
mn = ["J","F","M","A","M","J","J","A","S","O","N","D"]
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
ax = axes[0]
cols = [AMBER if i+1 in SUM else BLUE for i in range(12)]
ax.bar(range(12), by.N_rem, color=cols, zorder=2)
ax.plot(range(12), by.P_rem, color=INK, marker="o", ms=3.5, lw=1.4, zorder=3, label="Phosphorus")
ax.set_xticks(range(12)); ax.set_xticklabels(mn); ax.set_ylim(50, 95)
ax.set_ylabel("Removal efficiency (%)"); ax.set_title("Nitrogen removal collapses in summer")
ax.grid(True, axis="y"); clean(ax)
ax.legend(frameon=False, fontsize=7, loc="lower center")
ax.text(6.5, 62, "Jun–Aug", color=AMBER, fontsize=7.5, ha="center", weight="bold")

ax = axes[1]
ax.bar(range(12), by.BOD, color=cols, zorder=2)
ax.set_xticks(range(12)); ax.set_xticklabels(mn)
ax.set_ylabel("Effluent BOD$_5$ (mg L$^{-1}$)"); ax.set_title("Effluent BOD$_5$ more than doubles")
ax.grid(True, axis="y"); clean(ax)

ax = axes[2]
ax2 = ax.twinx()
ax.bar(range(12), by.N_out, color=cols, zorder=2)
ax2.plot(range(12), by["T"], color=INK, marker="s", ms=3.2, lw=1.4, zorder=3)
ax.set_xticks(range(12)); ax.set_xticklabels(mn)
ax.set_ylabel("Effluent N (mg L$^{-1}$)"); ax2.set_ylabel("Water temperature (°C)", color=INK)
ax.axhline(15, color=INK, ls=(0,(4,3)), lw=1.1)
ax.text(0.2, 15.6, "15 mg L$^{-1}$", fontsize=6.5, color=INK)
ax.set_title("Effluent N against temperature"); ax.grid(True, axis="y"); clean(ax, ("top",))
clean(ax2, ("top",))
fig.suptitle("Seasonal deterioration of biological treatment, 2015–2020 monthly means",
             fontsize=9.5, y=1.03)
fig.savefig(f"{OUT}/figA_seasonal.png"); plt.close(fig)

# ================= FIG B: per-parameter predictability =================
res = {}
for p in ["COD","BOD5","N","P","TSS"]:
    t = f"{p}_out"
    x = d[[t, *BASE]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    for L in (1,2,3,5,7,10): x[f"_lag{L}"] = x[t].shift(L)
    x["_roll"] = x[t].shift(1).rolling(10, min_periods=5).mean()
    x = x.dropna()
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    F = [c for c in x.columns if c != t]
    sc = StandardScaler().fit(tr[F])
    pr = LinearRegression().fit(sc.transform(tr[F]), tr[t]).predict(sc.transform(te[F]))
    res[p] = (r2_score(te[t], pr), r2_score(te[t], te["_roll"]))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ps = list(res); xx = np.arange(len(ps)); w = 0.37
ax.bar(xx-w/2, [res[p][0] for p in ps], w, color=BLUE, label="Model (linear regression)", zorder=2)
ax.bar(xx+w/2, [res[p][1] for p in ps], w, color=AMBER, label="Baseline (10-measurement mean)", zorder=2)
for i,p in enumerate(ps):
    for off,v in ((-w/2,res[p][0]), (w/2,res[p][1])):
        ax.text(i+off, v+0.015 if v>0 else 0.015, f"{v:.2f}", ha="center", fontsize=6.5, color=INK,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels(["COD","BOD$_5$","N","P","TSS"])
ax.set_ylabel("R² on measured test observations")
ax.set_title("Only nitrogen is meaningfully predictable")
ax.legend(frameon=False, fontsize=7); ax.grid(True, axis="y"); clean(ax)
fig.savefig(f"{OUT}/figB_predictability.png"); plt.close(fig)

# ================= FIG C: exceedance ROC =================
LIM = {"N_out":15.0, "BOD5_out":25.0, "P_out":2.0, "COD_out":50.0}
NICE = {"N_out":"Total N > 15 mg L$^{-1}$","BOD5_out":"BOD$_5$ > 25 mg L$^{-1}$",
        "P_out":"P > 2 mg L$^{-1}$","COD_out":"COD > 50 mg L$^{-1}$"}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
styles = [(BLUE,"-"), (AMBER,"-"), (BLUE,(0,(4,2))), (AMBER,(0,(4,2)))]
ax = axes[0]
for (t, lim), (col, ls) in zip(LIM.items(), styles):
    x = d[[t, *BASE]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    y = (x[t] > lim).astype(int)
    for L in (1,2,3,5,7,10):
        x[f"_lag{L}"] = x[t].shift(L); x[f"_exc{L}"] = y.shift(L)
    x["_roll"] = x[t].shift(1).rolling(10, min_periods=5).mean()
    x["_y"] = y; x = x.drop(columns=[t]).dropna()
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    F = [c for c in x.columns if c != "_y"]
    sc = StandardScaler().fit(tr[F])
    mdl = RandomForestClassifier(random_state=0, n_estimators=400, min_samples_leaf=3
                                 ).fit(sc.transform(tr[F]), tr["_y"])
    pp = mdl.predict_proba(sc.transform(te[F]))[:,1]
    fpr, tpr, _ = roc_curve(te["_y"], pp)
    ax.plot(fpr, tpr, color=col, ls=ls, lw=1.8, label=f"{NICE[t]}  AUC {roc_auc_score(te['_y'],pp):.2f}")
ax.plot([0,1],[0,1], color=MUTED, lw=1, ls=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Exceedance prediction, full model"); ax.legend(frameon=False, fontsize=6.8, loc="lower right")
ax.grid(True); clean(ax); ax.set_aspect("equal")

ax = axes[1]
for (t, lim), (col, ls) in zip([("N_out",15.0),("BOD5_out",25.0)], styles):
    x = d[[t, *BASE, "sin_year","cos_year"]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    y = (x[t] > lim).astype(int); x = x.drop(columns=[t])
    tr = x[x.index < "2019-02-01"]; te = x[(x.index >= "2019-04-15") & (x.index <= "2020-03-11")]
    sc = StandardScaler().fit(tr)
    mdl = GradientBoostingClassifier(random_state=0, n_estimators=250, max_depth=3
                                     ).fit(sc.transform(tr), y[tr.index])
    pp = mdl.predict_proba(sc.transform(te))[:,1]
    fpr, tpr, _ = roc_curve(y[te.index], pp)
    ax.plot(fpr, tpr, color=col, ls=ls, lw=1.8,
            label=f"{NICE[t]}  AUC {roc_auc_score(y[te.index],pp):.2f}")
ax.plot([0,1],[0,1], color=MUTED, lw=1, ls=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Influent chemistry + season only\n(no effluent history)")
ax.legend(frameon=False, fontsize=6.8, loc="lower right"); ax.grid(True); clean(ax); ax.set_aspect("equal")
fig.suptitle("Predicting regulatory exceedance, test period Apr 2019 – Mar 2020",
             fontsize=9.5, y=1.04)
fig.savefig(f"{OUT}/figC_exceedance_roc.png"); plt.close(fig)
print("wrote figA_seasonal, figB_predictability, figC_exceedance_roc")
