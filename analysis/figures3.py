import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from canon import d, INFL, OUTS, Q
OUT = os.environ.get("FIGDIR", str(_root / "figures")); os.makedirs(OUT,exist_ok=True)
BLUE,AMBER,INK,MUTED="#3B6FB6","#D97706","#1f2328","#6B7280"
plt.rcParams.update({"font.size":7.5,"axes.labelsize":7.5,"axes.titlesize":8.5,
 "axes.edgecolor":"#c9c9c6","axes.linewidth":0.8,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK,"figure.dpi":600,"savefig.dpi":600,
 "savefig.bbox":"tight","grid.color":"#e6e5e2","grid.linewidth":0.5})
def clean(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

# ---- Figure 1: all parameters through time ----
panels=[("Temperature_in",None,"Temperature (°C)"),("pH_in",None,"pH"),
        ("COD_in","COD_out","COD (mg L$^{-1}$)"),("BOD5_in","BOD5_out","BOD$_5$ (mg L$^{-1}$)"),
        ("N_in","N_out","Total N (mg L$^{-1}$)"),("P_in","P_out","Total P (mg L$^{-1}$)"),
        ("TSS_in","TSS_out","TSS (mg L$^{-1}$)"),("Q_flow",None,"Flow (m$^3$ d$^{-1}$)")]
fig,axes=plt.subplots(4,2,figsize=(7.4,8.2),sharex=True)
for ax,(a,b,lab) in zip(axes.ravel(),panels):
    s=d[a].dropna(); ax.plot(s.index,s.values,color=BLUE,lw=0.5,label="Influent" if b else None)
    if b:
        t=d[b].dropna(); ax.plot(t.index,t.values,color=AMBER,lw=0.5,label="Effluent")
        ax.legend(frameon=False,fontsize=6,loc="upper right",ncol=2)
    ax.set_ylabel(lab); ax.grid(True); clean(ax)
for ax in axes[-1]: ax.tick_params(axis="x",rotation=30)
fig.suptitle("Influent and effluent parameters through time, measured observations, 2015–2020",
             fontsize=9,y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_timeseries.png"); plt.close(fig)

# ---- Figure 4: violin plots ----
cols=[("COD_in","COD in"),("COD_out","COD out"),("BOD5_in","BOD$_5$ in"),("BOD5_out","BOD$_5$ out"),
      ("N_in","N in"),("N_out","N out"),("P_in","P in"),("P_out","P out"),
      ("TSS_in","TSS in"),("TSS_out","TSS out")]
fig,axes=plt.subplots(2,5,figsize=(9.2,4.2))
for ax,(c,lab) in zip(axes.ravel(),cols):
    v=d[c].dropna().values
    parts=ax.violinplot(v,showmedians=True,widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(AMBER if c.endswith("_out") else BLUE); pc.set_alpha(0.85)
    for k in ("cbars","cmins","cmaxes","cmedians"):
        if k in parts: parts[k].set_color(INK); parts[k].set_linewidth(0.9)
    ax.set_title(lab); ax.set_xticks([]); ax.grid(True,axis="y"); clean(ax)
    ax.set_ylabel("mg L$^{-1}$" if ax in axes[:,0] else "")
fig.suptitle("Distribution of influent (blue) and effluent (amber) parameters, measured observations",
             fontsize=9,y=1.01)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_violin.png"); plt.close(fig)
print("wrote fig1_timeseries, fig4_violin")
