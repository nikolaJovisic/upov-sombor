"""Correlation matrices and measured-vs-predicted, on the canonical frame."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from canon import d, INFL, OUTS, Q
OUT = os.environ.get("FIGDIR", str(_root / "figures")); os.makedirs(OUT,exist_ok=True)
BLUE,AMBER,INK,MUTED="#3B6FB6","#D97706","#1f2328","#6B7280"
DIV=LinearSegmentedColormap.from_list("d",[AMBER,"#f2f0ed",BLUE])
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":9,
 "axes.edgecolor":"#c9c9c6","axes.linewidth":0.8,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK,"figure.dpi":600,"savefig.dpi":600,
 "savefig.bbox":"tight","grid.color":"#e6e5e2","grid.linewidth":0.6})
NICE={"Temperature_in":"Temp$_{in}$","pH_in":"pH$_{in}$","COD_in":"COD$_{in}$",
      "BOD5_in":"BOD$_{5,in}$","N_in":"N$_{in}$","P_in":"P$_{in}$","TSS_in":"TSS$_{in}$"}
L=10
def build(t):
    x=d.copy(); tin=t.replace("_out","_in")
    for i in range(1,L+1):
        x[f"{t}_{i}"]=x[t].shift(i); x[f"{tin}_{i}"]=x[tin].shift(i)
    x["average"]=x[[f"{t}_{i}" for i in range(1,L+1)]].mean(axis=1)
    return x.dropna()
for t,tag,lab in [("COD_out","fig2","COD"),("BOD5_out","fig3","BOD$_5$")]:
    x=build(t); cols=INFL+[f"{t}_1","average",t]
    names=[NICE[c] for c in INFL]+[f"{lab}$_{{out,t-1}}$",f"{lab}$_{{out}}$ 10-meas. mean",f"{lab}$_{{out}}$"]
    C=x[cols].corr(); n=len(cols)
    fig,ax=plt.subplots(figsize=(5.4,4.6))
    im=ax.imshow(C,cmap=DIV,vmin=-1,vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names,rotation=45,ha="right"); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            v=C.iloc[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,
                    color="#ffffff" if abs(v)>0.55 else INK)
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1.4); ax.tick_params(which="minor",length=0)
    cb=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.03); cb.set_label("Pearson r",fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"Pearson correlation, {lab} predictors and effluent target\nmeasured observations only (n = {len(x)})",pad=8)
    fig.savefig(f"{OUT}/{tag}_correlation_{lab.replace('$','').replace('_','')}.png"); plt.close(fig)
    print(tag, "n=",len(x), "corr with target:", {c: round(float(C.loc[c,t]),2) for c in INFL})

fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
for ax,(t,lab) in zip(axes,[("COD_out","COD"),("BOD5_out","BOD$_5$")]):
    x=build(t); tin=t.replace("_out","_in")
    F=INFL+[c for c in x.columns if c.startswith((t+"_",tin+"_"))]+["average"]
    tr=x[x.index<"2019-02-01"]; te=x[(x.index>="2019-04-15")&(x.index<="2020-03-11")]
    sc=StandardScaler().fit(tr[INFL]); Xtr,Xte=tr[F].copy(),te[F].copy()
    Xtr[INFL]=sc.transform(tr[INFL]); Xte[INFL]=sc.transform(te[INFL])
    p=LinearRegression().fit(Xtr,tr[t]).predict(Xte); y=te[t].values
    lim=[0,max(y.max(),p.max())*1.05]
    ax.plot(lim,lim,color=INK,lw=1.2,zorder=1,label="1:1 line")
    ax.scatter(y,p,s=18,color=BLUE,alpha=.75,lw=.5,edgecolor="white",zorder=2,label="Linear regression")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"Measured effluent {lab} (mg L$^{{-1}}$)")
    ax.set_ylabel(f"Predicted effluent {lab} (mg L$^{{-1}}$)")
    ax.grid(True,lw=.6,alpha=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(f"{lab}  (n = {len(y)})"); ax.legend(frameon=False,fontsize=7,loc="upper left")
fig.suptitle("Measured versus predicted effluent concentration, test period 15 Apr 2019 – 11 Mar 2020",fontsize=9,y=1.02)
fig.savefig(f"{OUT}/fig5_measured_vs_predicted.png"); plt.close(fig)
print("done")
