"""Final horizon results + the skill-decay figure. Writes horizon_numbers.json."""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, torch, random
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import WaterNet
from canon import d, INFL, OUTS, Q

OUT = os.environ.get("FIGDIR", str(_root / "figures")); os.makedirs(OUT, exist_ok=True)
BLUE, AMBER, INK, MUTED = "#3B6FB6", "#D97706", "#1f2328", "#6B7280"
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":9,
 "axes.edgecolor":"#c9c9c6","axes.linewidth":0.8,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK,"figure.dpi":600,"savefig.dpi":600,
 "savefig.bbox":"tight","grid.color":"#e6e5e2","grid.linewidth":0.6})
rng = np.random.default_rng(0); L = 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"
HS = [1,2,3,5,7,10]

def build(t,h):
    x=d.copy(); tin=t.replace("_out","_in")
    flat=list(INFL)+["sin_year","cos_year"]; seq=[]
    for i in range(0,L):
        x[f"y_{i}"]=x[t].shift(i); x[f"i_{i}"]=x[tin].shift(i); seq+=[f"y_{i}",f"i_{i}"]
    x["roll"]=x[t].rolling(L,min_periods=L).mean(); flat.append("roll")
    x["persist"]=x[t]; x["target"]=x[t].shift(-h)
    return x.dropna(subset=flat+seq+["target","persist"]), flat, seq

def boot(y,pm,pb,n=6000):
    y,pm,pb=map(np.asarray,(y,pm,pb)); idx=rng.integers(0,len(y),(n,len(y)))
    Y,M,B=y[idx],pm[idx],pb[idx]; sst=((Y-Y.mean(1,keepdims=True))**2).sum(1)
    dd=(((Y-B)**2).sum(1)-((Y-M)**2).sum(1))/sst
    return round(float(np.percentile(dd,2.5)),3), round(float(np.percentile(dd,97.5)),3)

def lstm_pred(tr,te,flat,seq,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    F=flat; sc=StandardScaler().fit(tr[F])
    def T(df):
        return ([torch.tensor(sc.transform(df[F]),dtype=torch.float32),
                 torch.tensor(df[seq].values.reshape(len(df),L,2),dtype=torch.float32)],
                torch.tensor(df[["target"]].values,dtype=torch.float32))
    cut=int(0.8*len(tr)); tra,va=tr.iloc[:cut],tr.iloc[cut+L:]
    (Xtr,ytr),(Xva,yva),(Xte,_)=T(tra),T(va),T(te)
    net=WaterNet(features_size=len(F),seq_length=L,seq_channels=2,output_size=1,mode="lstm")
    opt=optim.Adam(net.parameters(),lr=0.005); sch=ExponentialLR(opt,gamma=0.9); crit=nn.MSELoss()
    dl=DataLoader(TensorDataset(*Xtr,ytr),batch_size=8,shuffle=False,drop_last=True)
    f=lambda xs: net((xs[0],xs[1]))
    best,state=np.inf,None
    for _ in range(30):
        net.train()
        for b in dl:
            *xb,yb=b; opt.zero_grad(); crit(f(xb),yb).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl=crit(f(Xva),yva).item()
            if vl<best: best,state=vl,{k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return f(Xte).numpy().ravel()

RES={}
for t in ["N_out","COD_out","BOD5_out"]:
    RES[t]={}
    for h in HS:
        x,flat,seq=build(t,h); tr=x[x.index<TR_END]; te=x[(x.index>=TE_A)&(x.index<=TE_B)]
        y=te["target"].values; F=flat+seq
        sc=StandardScaler().fit(tr[F])
        rf=RandomForestRegressor(random_state=0,n_estimators=300,min_samples_leaf=3
           ).fit(sc.transform(tr[F]),tr["target"]).predict(sc.transform(te[F]))
        lr=LinearRegression().fit(sc.transform(tr[F]),tr["target"]).predict(sc.transform(te[F]))
        ls=[lstm_pred(tr,te,flat,seq,s) for s in range(3)]
        lsm=np.mean(ls,axis=0)
        RES[t][h]=dict(n=int(len(te)),
          rf=round(float(r2_score(y,rf)),3), lr=round(float(r2_score(y,lr)),3),
          lstm=round(float(r2_score(y,lsm)),3),
          lstm_sd=round(float(np.std([r2_score(y,p) for p in ls],ddof=1)),3),
          persist=round(float(r2_score(y,te["persist"])),3),
          roll=round(float(r2_score(y,te["roll"])),3),
          rf_vs_persist=boot(y,rf,te["persist"].values), rf_vs_roll=boot(y,rf,te["roll"].values),
          lstm_vs_persist=boot(y,lsm,te["persist"].values), lstm_vs_roll=boot(y,lsm,te["roll"].values))
        print(t,h,RES[t][h]["rf"],RES[t][h]["lstm"],RES[t][h]["persist"],RES[t][h]["roll"])
json.dump(RES,open(Path(__file__).with_name("horizon_numbers.json"),"w"),indent=1)

# ---------------- Figure: skill decay ----------------
fig,axes=plt.subplots(1,3,figsize=(9.4,3.2),sharey=True)
for ax,(t,lab) in zip(axes,[("N_out","Total nitrogen"),("BOD5_out","BOD$_5$"),("COD_out","COD")]):
    hs=HS
    ax.plot(hs,[RES[t][h]["rf"] for h in hs],color=BLUE,marker="o",ms=4,lw=1.8,label="Random forest")
    ax.plot(hs,[RES[t][h]["lstm"] for h in hs],color=BLUE,marker="s",ms=3.5,lw=1.4,
            ls=(0,(4,2)),label="LSTM")
    ax.plot(hs,[RES[t][h]["roll"] for h in hs],color=AMBER,marker="^",ms=4,lw=1.8,
            label="Baseline: rolling mean")
    ax.plot(hs,[RES[t][h]["persist"] for h in hs],color=AMBER,marker="v",ms=3.5,lw=1.4,
            ls=(0,(4,2)),label="Baseline: persistence")
    ax.axhline(0,color=MUTED,lw=0.8)
    ax.set_xticks(hs); ax.set_xlabel("Forecast horizon (measurements ahead)")
    ax.set_title(lab); ax.grid(True); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("R² on held-out test period")
axes[0].legend(frameon=False,fontsize=6.8,loc="upper right")
fig.suptitle("Model skill persists with forecast horizon; trivial baselines do not",fontsize=9.5,y=1.03)
fig.savefig(f"{OUT}/figD_horizon.png"); plt.close(fig)
print("wrote figD_horizon.png")
