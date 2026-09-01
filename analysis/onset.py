"""Can the plant be warned before a nitrogen breach, rather than during one?

explain.py showed the deployed forest never fires on a breach that follows a compliant
measurement. This asks whether that is fixable, and separates three different answers.

  A. Move the threshold. Same model, different operating point. Not a better model,
     but it is the honest first question: is the ranking any good on these days?
  B. Train for the job. Restrict to days where the previous measurement was compliant
     and predict a breach today, so the model cannot win by carrying yesterday forward.
  C. Forecast further out. Predict a breach five and ten measurements ahead.

Every model is judged against the trivial alternative an operator already has: how close
the last measurement was to the limit. Beating that is the whole test.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
_root = next(p for p in Path(__file__).resolve().parents if (p / "data.csv").exists())
sys.path[:0] = [str(_root), str(_root / "analysis")]

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from canon import d as CANON, INFL, Q

L, LIMIT = 10, 15.0
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"
rng = np.random.default_rng(0)
OUT = {}

FIGDIR = os.environ.get("FIGDIR", str(_root / "figures"))
os.makedirs(FIGDIR, exist_ok=True)
BLUE, AMBER, INK, MUTED = "#3B6FB6", "#D97706", "#1f2328", "#6B7280"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "axes.edgecolor": "#c9c9c6", "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "figure.dpi": 600, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "grid.color": "#e6e5e2", "grid.linewidth": 0.6})


def build(horizon=0):
    """Lagged frame with the target `horizon` measurements ahead. horizon 0 is same day."""
    x = CANON.copy()
    for i in range(1, L + 1):
        x[f"N_out_{i}"] = x["N_out"].shift(i)
        x[f"N_in_{i}"] = x["N_in"].shift(i)
    x["N_out_mean10"] = x["N_out"].shift(1).rolling(L, min_periods=5).mean()
    x["N_out_trend3"] = x["N_out"].shift(1) - x["N_out"].shift(4)
    x["gap_to_limit"] = LIMIT - x["N_out"].shift(1)
    x["removal_1"] = 100 * (1 - x["N_out"].shift(1) / x["N_in"].shift(1))
    breach = (x["N_out"] > LIMIT).astype(int)
    for i in range(1, L + 1):
        x[f"exc{i}"] = breach.shift(i)
    x["excroll"] = breach.shift(1).rolling(L, min_periods=5).mean()
    x["_y"] = breach.shift(-horizon)
    x["_prev_ok"] = (breach.shift(0) == 0).astype(int) if horizon else (breach.shift(1) == 0).astype(int)
    return x.dropna()


FEATURES = (list(INFL) + [Q, "sin_year", "cos_year"]
            + [f"N_out_{i}" for i in range(1, L + 1)]
            + [f"N_in_{i}" for i in range(1, L + 1)]
            + ["N_out_mean10", "N_out_trend3", "gap_to_limit", "removal_1", "excroll"])

MODELS = {
    "logistic": lambda: LogisticRegression(max_iter=4000, class_weight="balanced"),
    "random forest": lambda: RandomForestClassifier(random_state=0, n_estimators=600,
                                                    min_samples_leaf=2,
                                                    class_weight="balanced"),
    "gradient boosting": lambda: GradientBoostingClassifier(random_state=0, n_estimators=300,
                                                            max_depth=3),
}


def auc_ci(y, s, n=4000):
    a = []
    y, s = np.asarray(y), np.asarray(s)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            a.append(roc_auc_score(y[i], s[i]))
    return round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)


def evaluate(name, tr, te, feats):
    """Fit each model, score against the proximity baseline, report the best."""
    ytr, yte = tr["_y"].values, te["_y"].values
    if len(np.unique(yte)) < 2 or ytr.sum() < 10:
        print(f"  {name}: not enough events"); return None
    # Trivial alternative: how close the last measurement already was to the limit.
    base_auc = roc_auc_score(yte, -te["gap_to_limit"].values)

    # Pick the model on a validation period inside training, never on the test set.
    cut = int(0.8 * len(tr))
    fit, val = tr.iloc[:cut], tr.iloc[cut:]
    scv = StandardScaler().fit(fit[feats])
    chosen, best_val = None, -np.inf
    for mname, make in MODELS.items():
        if len(np.unique(val["_y"].values)) < 2:
            chosen = list(MODELS)[0]; break
        m = make().fit(scv.transform(fit[feats]), fit["_y"])
        v = roc_auc_score(val["_y"], m.predict_proba(scv.transform(val[feats]))[:, 1])
        if v > best_val:
            chosen, best_val = mname, v
    mname = chosen

    # Refit the chosen model on the whole training period, then score the test period once.
    sc = StandardScaler().fit(tr[feats])
    m = MODELS[mname]().fit(sc.transform(tr[feats]), ytr)
    p = m.predict_proba(sc.transform(te[feats]))[:, 1]
    a = roc_auc_score(yte, p)
    lo, hi = auc_ci(yte, p)
    # Operating point that catches two thirds of events, and what it costs.
    order = np.sort(p[yte == 1])
    thr = float(order[max(0, int(len(order) * 0.33))])
    pred = (p >= thr).astype(int)
    res = dict(selected_on="validation period inside training", n_train=int(len(tr)), n_test=int(len(te)), events=int(yte.sum()),
               rate=round(100 * float(yte.mean()), 1), model=mname,
               auc=round(float(a), 3), auc_ci=[lo, hi],
               baseline_auc=round(float(base_auc), 3),
               beats_baseline=bool(lo > base_auc),
               at_recall_67=dict(threshold=round(thr, 3),
                                 precision=round(float(precision_score(yte, pred, zero_division=0)), 2),
                                 recall=round(float(recall_score(yte, pred)), 2),
                                 alarms_per_100_days=round(100 * float(pred.mean()), 1)))
    flag = "beats it" if res["beats_baseline"] else "does not beat it"
    print(f"  {name}: {mname}, AUC {a:.3f} [{lo:.3f},{hi:.3f}] vs proximity baseline "
          f"{base_auc:.3f} -> {flag}")
    print(f"      at 67% recall: precision {res['at_recall_67']['precision']:.2f}, "
          f"{res['at_recall_67']['alarms_per_100_days']:.0f} alarms per 100 days, "
          f"{res['events']} events in {res['n_test']} test days")
    return res


def deployed_frame():
    """The exact construction in final.py, so section A speaks about the published model."""
    x = CANON.drop(columns=["sin_year", "cos_year"]).copy()
    for i in range(1, L + 1):
        x[f"N_out_{i}"] = x["N_out"].shift(i)
        x[f"N_in_{i}"] = x["N_in"].shift(i)
    x["average"] = x[[f"N_out_{i}" for i in range(1, L + 1)]].mean(axis=1)
    x = x.dropna()
    breach = (x["N_out"] > LIMIT).astype(int)
    for i in range(1, L + 1):
        x[f"exc{i}"] = breach.shift(i)
    x["excroll"] = breach.shift(1).rolling(L, min_periods=5).mean()
    x["_y"] = breach
    x = x.dropna()
    outs = ["COD_out", "BOD5_out", "N_out", "P_out", "TSS_out"]
    feats = [c for c in x.columns if c not in ("_y",) + tuple(outs)]
    return x, feats


print("A. THE DEPLOYED MODEL, JUDGED ONLY ON ONSET DAYS")
xd, fd = deployed_frame()
trd, ted = xd[xd.index < TR_END], xd[(xd.index >= TE_A) & (xd.index <= TE_B)]
scd = StandardScaler().fit(trd[fd])
dep = RandomForestClassifier(random_state=0, n_estimators=400,
                             min_samples_leaf=3).fit(scd.transform(trd[fd]), trd["_y"])
pdep = dep.predict_proba(scd.transform(ted[fd]))[:, 1]
print(f"  reproduces the published model: AUC {roc_auc_score(ted['_y'], pdep):.3f} on all days")

mask = ted["exc1"].values == 0
yo, po = ted["_y"].values[mask], pdep[mask]
gap_o = LIMIT - ted["N_out_1"].values[mask]
base_o = roc_auc_score(yo, -gap_o)
oa = roc_auc_score(yo, po)
olo, ohi = auc_ci(yo, po)
print(f"  onset days: {int(yo.sum())} breaches in {int(mask.sum())} days")
print(f"  ranking on those days: AUC {oa:.3f} [{olo:.3f},{ohi:.3f}] "
      f"vs proximity baseline {base_o:.3f}")
print(f"  highest probability it gives any of them: {po[yo == 1].max():.2f}, "
      f"so nothing fires at 0.5")

rows = []
for thr in (0.5, 0.4, 0.35, 0.3, 0.25, 0.2):
    pred = (po >= thr).astype(int)
    rows.append(dict(threshold=thr,
                     recall=round(float(recall_score(yo, pred)), 2),
                     precision=round(float(precision_score(yo, pred, zero_division=0)), 2),
                     alarms_per_100_days=round(100 * float(pred.mean()), 1)))
print(f"  {'threshold':>10}{'recall':>9}{'precision':>11}{'alarms/100d':>13}")
for r in rows:
    print(f"  {r['threshold']:10.2f}{r['recall']:9.2f}{r['precision']:11.2f}"
          f"{r['alarms_per_100_days']:13.1f}")
print(f"  (base rate on these days is {100 * yo.mean():.0f}%)")
OUT["A_deployed_on_onset"] = dict(auc=round(float(oa), 3), auc_ci=[olo, ohi],
                                  proximity_baseline_auc=round(float(base_o), 3),
                                  n_days=int(mask.sum()), events=int(yo.sum()),
                                  base_rate_pct=round(100 * float(yo.mean()), 1),
                                  max_probability=round(float(po[yo == 1].max()), 3),
                                  thresholds=rows)

print("\nB. TRAINED FOR THE JOB: only days that follow a compliant measurement")
xb = build(0)
xb = xb[xb["_prev_ok"] == 1]
OUT["B_onset_model"] = evaluate("onset model", xb[xb.index < TR_END],
                                xb[(xb.index >= TE_A) & (xb.index <= TE_B)], FEATURES)

print("\nC. FORECAST FURTHER OUT, all days")
for h in (5, 10):
    xh = build(h)
    OUT[f"C_horizon_{h}"] = evaluate(f"breach {h} measurements ahead",
                                     xh[xh.index < TR_END],
                                     xh[(xh.index >= TE_A) & (xh.index <= TE_B)], FEATURES)

# ---------------------------------------------------------------------------------
# The operating point, drawn. This is the figure the deployment paragraph needs.
# ---------------------------------------------------------------------------------
grid = np.linspace(0.05, 0.60, 60)
rec = [recall_score(yo, (po >= t).astype(int)) for t in grid]
pre = [precision_score(yo, (po >= t).astype(int), zero_division=np.nan) for t in grid]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(grid, rec, color=BLUE, lw=1.8, label="Recall, share of onset breaches caught")
ax.plot(grid, pre, color=AMBER, lw=1.8, label="Precision, share of alarms that are real")
ax.axhline(float(yo.mean()), color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.text(0.585, float(yo.mean()) + 0.015, "base rate", fontsize=6.8, color=MUTED, ha="right")
ax.axvline(0.5, color=INK, lw=0.9, ls=(0, (2, 2)))
ax.text(0.5, 1.02, "published threshold", fontsize=6.8, color=INK, ha="center")
ax.axvline(0.35, color=BLUE, lw=0.9, ls=(0, (2, 2)))
ax.text(0.35, 1.02, "proposed", fontsize=6.8, color=BLUE, ha="center")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Share")
ax.set_xlim(0.05, 0.60); ax.set_ylim(0, 1.08)
ax.grid(True); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.set_title("On days following a compliant measurement, 0.5 fires on nothing", fontsize=9)
fig.savefig(f"{FIGDIR}/figI_onset_thresholds.png")
plt.close(fig)
print("wrote figI_onset_thresholds.png")

json.dump(OUT, open(Path(__file__).with_name("onset_numbers.json"), "w"), indent=1)
print("wrote onset_numbers.json")
