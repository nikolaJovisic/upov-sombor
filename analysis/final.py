"""Single source of truth: every number reported in the manuscript, from one frame.

CANONICAL CLEANING (applied identically to all analyses):
  drop duplicate '(2)' records, drop rows with missing influent/effluent values,
  flow in (1000, 30000) m3/day, no zero values, effluent COD < 125 and BOD5 < 180,
  influent COD < 2000, BOD5 < 900, N < 200, P < 35, TSS < 5000, average duplicate dates.
CANONICAL SPLIT: train < 2019-02-01, test 2019-04-15..2020-03-11, 74-day embargo.
"""
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd, torch, random
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error as mape
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from dataset import load, format_date, format_strings
from model import WaterNet

INFL = ["Temperature_(C˚)_in","pH_in","COD_(mg/l)_in","BOD5_(mg/l)_in",
        "N_(mg/l)_in","P_(mg/l)_in","TSS_(mg/l)_in"]
OUTS = ["COD_(mg/l)_out","BOD5_(mg/l)_out","N_(mg/l)_out","P_(mg/l)_out","TSS_(mg/l)_out"]
QC, L = "Q_(m3/day)_flow", 10
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"
rng = np.random.default_rng(0)
OUT = {}

# ---------------- canonical frame ----------------
steps = []
d = load(); steps.append(("records in source file", len(d)))
d = d.dropna(subset=["Date"]); d = d[~d["Date"].str.endswith("(2)")]
steps.append(("after duplicate removal", len(d)))
d = d.dropna(subset=INFL).dropna(subset=OUTS)
steps.append(("after missing-value removal", len(d)))
d = d.map(lambda x: format_date(x) if isinstance(x, str) else x)
d["Date"] = pd.to_datetime(d["Date"], format="%d.%m.%Y")
d = d.set_index("Date").map(format_strings).loc[:, [*INFL, *OUTS, QC]]
d = d[(d[QC] > 1000) & (d[QC] < 30000)]; steps.append(("after flow plausibility filter", len(d)))
d = d[~(d[[*INFL, *OUTS]] == 0).any(axis=1)]; steps.append(("after zero-value removal", len(d)))
d = d[(d["COD_(mg/l)_out"] < 125) & (d["BOD5_(mg/l)_out"] < 180)]
steps.append(("after effluent plausibility filters", len(d)))
d = d[(d["COD_(mg/l)_in"] < 2000) & (d["BOD5_(mg/l)_in"] < 900) & (d["N_(mg/l)_in"] < 200) &
      (d["P_(mg/l)_in"] < 35) & (d["TSS_(mg/l)_in"] < 5000)]
steps.append(("after influent plausibility filters", len(d)))
D = d.groupby("Date").mean().sort_index()
steps.append(("distinct measurement days", len(D)))
OUT["cascade"] = steps
gaps = pd.Series(D.index).diff().dt.days.dropna()
OUT["sampling"] = dict(median=float(gaps.median()), mean=round(float(gaps.mean()),2),
                       within2=round(100*float((gaps<=2).mean()),1),
                       span_days=int((D.index.max()-D.index.min()).days+1),
                       first=str(D.index.min().date()), last=str(D.index.max().date()))

# ---------------- descriptive ----------------
OUT["describe"] = {c: {k: round(float(v),1) for k,v in D[c].describe()[["mean","std","min","25%","50%","75%","max"]].items()}
                   for c in [*INFL, *OUTS, QC]}
OUT["removal"] = {p: round(100*(1-D[f"{p}_(mg/l)_out" if p!="Temperature" else p].mean()
                               /D[f"{p}_(mg/l)_in"].mean()),1) for p in ["COD","BOD5","N","P","TSS"]}
OUT["attenuation"] = {}
for p in ["COD","TSS","BOD5","P","N"]:
    i,o = f"{p}_(mg/l)_in", f"{p}_(mg/l)_out"
    OUT["attenuation"][p] = dict(sd_in=round(float(D[i].std()),1), sd_out=round(float(D[o].std()),1),
        atten=round(float(D[i].std()/D[o].std()),1), r=round(float(D[i].corr(D[o])),2),
        removal=round(100*(1-D[o].mean()/D[i].mean()),1))

# ---------------- seasonal ----------------
m = D.copy(); m["N_rem"]=100*(1-m["N_(mg/l)_out"]/m["N_(mg/l)_in"])
m["P_rem"]=100*(1-m["P_(mg/l)_out"]/m["P_(mg/l)_in"])
su, rs = m[m.index.month.isin([6,7,8])], m[~m.index.month.isin([6,7,8])]
OUT["seasonal"] = dict(
 n_rem=(round(su.N_rem.mean(),1), round(rs.N_rem.mean(),1)),
 p_rem=(round(su.P_rem.mean(),1), round(rs.P_rem.mean(),1)),
 n_out=(round(su["N_(mg/l)_out"].mean(),1), round(rs["N_(mg/l)_out"].mean(),1)),
 bod_out=(round(su["BOD5_(mg/l)_out"].mean(),1), round(rs["BOD5_(mg/l)_out"].mean(),1)),
 cod_out=(round(su["COD_(mg/l)_out"].mean(),1), round(rs["COD_(mg/l)_out"].mean(),1)),
 n_in=(round(su["N_(mg/l)_in"].mean(),1), round(rs["N_(mg/l)_in"].mean(),1)),
 flow=(round(su[QC].mean()), round(rs[QC].mean())),
 p_nrem=float(stats.mannwhitneyu(su.N_rem, rs.N_rem)[1]),
 p_bod=float(stats.mannwhitneyu(su["BOD5_(mg/l)_out"], rs["BOD5_(mg/l)_out"])[1]),
 exc15=(round(100*float((su["N_(mg/l)_out"]>15).mean()),1), round(100*float((rs["N_(mg/l)_out"]>15).mean()),1)))
OUT["seasonal_by_year"] = {int(y): (round(g[g.index.month.isin([6,7,8])].N_rem.mean(),1),
                                    round(g[~g.index.month.isin([6,7,8])].N_rem.mean(),1))
                           for y,g in m.groupby(m.index.year) if len(g)>40}

# ---------------- modelling helpers ----------------
def frame(target):
    x = D.copy(); tin = target[:-3]+"in"
    seq=[]
    for i in range(1, L+1):
        x[f"{target}_{i}"]=x[target].shift(i); x[f"{tin}_{i}"]=x[tin].shift(i)
        seq += [f"{target}_{i}", f"{tin}_{i}"]
    x["average"]=x[[f"{target}_{i}" for i in range(1,L+1)]].mean(axis=1)
    x=x.dropna()
    return x, seq

def split(x):
    return x[x.index<TR_END], x[(x.index>=TE_A)&(x.index<=TE_B)]

def metrics(y,p): return dict(r2=round(float(r2_score(y,p)),3),
    rmse=round(float(np.sqrt(mean_squared_error(y,p))),2), mape=round(float(mape(y,p)),3))

def boot(y, pm, pb, n=10000):
    y,pm,pb = map(np.asarray,(y,pm,pb)); idx=rng.integers(0,len(y),(n,len(y)))
    Y,M,B = y[idx],pm[idx],pb[idx]; sst=((Y-Y.mean(1,keepdims=True))**2).sum(1)
    diff=(((Y-B)**2).sum(1)-((Y-M)**2).sum(1))/sst
    return round(float(np.percentile(diff,2.5)),3), round(float(np.percentile(diff,97.5)),3)

# ---------------- regression, all five parameters ----------------
OUT["regression"] = {}
for target in OUTS:
    x, seq = frame(target); feats = INFL + seq + ["average"]
    tr, te = split(x)
    sc = StandardScaler().fit(tr[INFL])
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    Xtr[INFL]=sc.transform(tr[INFL]); Xte[INFL]=sc.transform(te[INFL])
    lr = LinearRegression().fit(Xtr, tr[target]); pred = lr.predict(Xte)
    base = te["average"].values
    r = dict(n_train=len(tr), n_test=len(te), n_model=len(x),
             lr=metrics(te[target],pred), roll=metrics(te[target],base),
             prev=metrics(te[target],te[f"{target}_1"]),
             trmean=metrics(te[target],np.full(len(te),tr[target].mean())),
             influent_only=metrics(te[target],
                 LinearRegression().fit(Xtr[INFL],tr[target]).predict(Xte[INFL])),
             ci_lr_vs_base=boot(te[target],pred,base))
    sd=Xtr.std(0).replace(0,np.nan); sh=100*(lr.coef_*sd).abs()/(lr.coef_*sd).abs().sum()
    lag=[f for f in feats if target in f]+["average"]
    r["lagged_pct"]=round(float(sh[[f for f in lag if f in feats]].sum()),1)
    OUT["regression"][target.split("_")[0]] = r

# ---------------- neural models, COD and BOD5 ----------------
def run_nn(target, mode, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x, seq = frame(target); use_seq = mode!="mlp"
    flat = INFL + ["average"] + ([] if use_seq else seq)
    tr_all, te = split(x); sc = StandardScaler().fit(tr_all[INFL])
    def T(df):
        X=df[flat].copy(); X[INFL]=sc.transform(df[INFL])
        xs=[torch.tensor(X.values,dtype=torch.float32)]
        if use_seq: xs.append(torch.tensor(df[seq].values.reshape(len(df),L,2),dtype=torch.float32))
        return xs, torch.tensor(df[[target]].values,dtype=torch.float32)
    cut=int(0.8*len(tr_all)); tr,va = tr_all.iloc[:cut], tr_all.iloc[cut+L:]
    (Xtr,ytr),(Xva,yva),(Xte,yte)=T(tr),T(va),T(te)
    net=WaterNet(features_size=len(flat),seq_length=L,seq_channels=2,output_size=1,mode=mode)
    opt=optim.Adam(net.parameters(),lr=0.005); sch=ExponentialLR(opt,gamma=0.9); crit=nn.MSELoss()
    dl=DataLoader(TensorDataset(*Xtr,ytr),batch_size=8,shuffle=False,drop_last=True)
    f=lambda xs: net((xs[0],xs[1])) if use_seq else net(xs[0])
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
    with torch.no_grad(): p=f(Xte).numpy().ravel()
    return yte.numpy().ravel(), p

OUT["neural"]={}
for target in ["COD_(mg/l)_out","BOD5_(mg/l)_out"]:
    for mode in ["mlp","conv","lstm"]:
        preds=[];
        for s in range(5):
            y,p=run_nn(target,mode,s); preds.append(p)
        r2s=[r2_score(y,p) for p in preds]
        x,_=frame(target); _,te=split(x); base=te["average"].values
        OUT["neural"][f"{target.split('_')[0]}_{mode}"]=dict(
            r2=round(float(np.mean(r2s)),3), sd=round(float(np.std(r2s,ddof=1)),3),
            rmse=round(float(np.mean([np.sqrt(mean_squared_error(y,p)) for p in preds])),2),
            mape=round(float(np.mean([mape(y,p) for p in preds])),3),
            ci_vs_base=boot(y,np.mean(preds,axis=0),base))

# ---------------- exceedance classification ----------------
LIM={"N_(mg/l)_out":15.0,"BOD5_(mg/l)_out":25.0,"P_(mg/l)_out":2.0,"COD_(mg/l)_out":50.0}
OUT["exceed"]={}
for target,lim in LIM.items():
    x,seq=frame(target); y=(x[target]>lim).astype(int)
    for i in range(1,L+1): x[f"exc{i}"]=y.shift(i)
    x["excroll"]=y.shift(1).rolling(10,min_periods=5).mean(); x["_y"]=y
    x=x.dropna(); tr,te=split(x)
    F=[c for c in x.columns if c not in ("_y",)+tuple(OUTS)]
    sc=StandardScaler().fit(tr[F])
    best=None
    for nm,mdl in [("logistic",LogisticRegression(max_iter=2000)),
                   ("random forest",RandomForestClassifier(random_state=0,n_estimators=400,min_samples_leaf=3)),
                   ("gradient boosting",GradientBoostingClassifier(random_state=0,n_estimators=250,max_depth=3))]:
        mdl.fit(sc.transform(tr[F]),tr["_y"]); pp=mdl.predict_proba(sc.transform(te[F]))[:,1]
        a=roc_auc_score(te["_y"],pp)
        if best is None or a>best[1]: best=(nm,a,pp)
    nm,a,pp=best
    bl=max(roc_auc_score(te["_y"],te["excroll"]), roc_auc_score(te["_y"],te["average"]),
           roc_auc_score(te["_y"],te["exc1"]))
    aucs=[]
    for _ in range(4000):
        i=rng.integers(0,len(te),len(te))
        if len(np.unique(te["_y"].values[i]))>1: aucs.append(roc_auc_score(te["_y"].values[i],pp[i]))
    tn,fp,fn,tp=confusion_matrix(te["_y"],(pp>0.5).astype(int)).ravel()
    OUT["exceed"][target.split("_")[0]]=dict(model=nm,auc=round(float(a),3),
        ci=(round(float(np.percentile(aucs,2.5)),3),round(float(np.percentile(aucs,97.5)),3)),
        baseline=round(float(bl),3), precision=round(tp/(tp+fp),2) if tp+fp else 0,
        recall=round(tp/(tp+fn),2) if tp+fn else 0, tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),
        n_test=int(len(te)), rate=round(100*float(te["_y"].mean()),1))

json.dump(OUT, open(Path(__file__).with_name("final_numbers.json"),"w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:400])
print("\nWROTE final_numbers.json")
