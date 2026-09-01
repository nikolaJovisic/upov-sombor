"""The claims in the paper, asserted against the pipeline.

Fast checks only. Nothing here trains a neural network, so the suite runs in CI on
every push. Point estimates move slightly between scikit-learn releases, so the
model assertions use tolerances while the data assertions are exact.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]

from canon import d, INFL, OUTS, Q  # noqa: E402

SUMMER = [6, 7, 8]
TR_END, TE_A, TE_B = "2019-02-01", "2019-04-15", "2020-03-11"


def removal(p):
    return 100 * (1 - d[f"{p}_out"] / d[f"{p}_in"])


def season(s):
    return s[s.index.month.isin(SUMMER)], s[~s.index.month.isin(SUMMER)]


def test_measured_days():
    """Cleaning keeps 1,017 measurement days, spanning 2015 to 2020."""
    assert len(d) == 1017
    assert str(d.index.min().date()) == "2015-01-08"
    assert str(d.index.max().date()) == "2020-07-27"


def test_no_gaps_were_filled():
    """Every row is a day the plant sampled. Half the calendar days are absent."""
    calendar_days = (d.index.max() - d.index.min()).days + 1
    assert len(d) / calendar_days < 0.55
    assert not d[[*INFL, *OUTS, Q]].isna().any().any()


@pytest.mark.parametrize("p,summer,rest", [("N", 67.4, 79.4)])
def test_summer_nitrogen_deterioration(p, summer, rest):
    """Nitrogen removal falls by twelve points in summer."""
    su, rs = season(removal(p))
    assert round(su.mean(), 1) == summer
    assert round(rs.mean(), 1) == rest
    assert stats.mannwhitneyu(su, rs)[1] < 1e-15


def test_solids_removal_is_unaffected():
    """The mechanism argument: settling does not fail, so this is not a solids problem."""
    su, rs = season(removal("TSS"))
    assert stats.mannwhitneyu(su, rs)[1] > 0.05


def test_summer_effect_holds_in_most_years():
    """Present in five of the six monitored years."""
    rem = removal("N")
    worse = 0
    for _, g in rem.groupby(rem.index.year):
        su, rs = season(g)
        if len(g) > 40 and su.mean() < rs.mean():
            worse += 1
    assert worse >= 5


def test_effluent_bod_rises_in_summer():
    su, rs = season(d["BOD5_out"])
    assert round(su.mean(), 1) == 62.2
    assert round(rs.mean(), 1) == 26.7


def horizon_fit(target, h, lags=10):
    x = d.copy()
    tin = target.replace("_out", "_in")
    flat = list(INFL) + ["sin_year", "cos_year"]
    seq = []
    for i in range(lags):
        x[f"y_{i}"] = x[target].shift(i)
        x[f"i_{i}"] = x[tin].shift(i)
        seq += [f"y_{i}", f"i_{i}"]
    x["roll"] = x[target].rolling(lags, min_periods=lags).mean()
    flat.append("roll")
    x["persist"] = x[target]
    x["target"] = x[target].shift(-h)
    x = x.dropna(subset=flat + seq + ["target", "persist"])
    tr = x[x.index < TR_END]
    te = x[(x.index >= TE_A) & (x.index <= TE_B)]
    F = flat + seq
    sc = StandardScaler().fit(tr[F])
    rf = RandomForestRegressor(random_state=0, n_estimators=300, min_samples_leaf=3)
    rf.fit(sc.transform(tr[F]), tr["target"])
    y = te["target"].values
    pred = rf.predict(sc.transform(te[F]))
    return dict(y=y, pred=pred, rf=r2_score(y, pred),
                roll=r2_score(y, te["roll"].values),
                persist=r2_score(y, te["persist"].values),
                roll_pred=te["roll"].values, persist_pred=te["persist"].values)


def advantage_ci(y, model, baseline, n=4000, seed=0):
    """Bootstrap interval on the R2 gained over a baseline, as in the paper."""
    rng = np.random.default_rng(seed)
    y, model, baseline = map(np.asarray, (y, model, baseline))
    idx = rng.integers(0, len(y), (n, len(y)))
    Y, M, B = y[idx], model[idx], baseline[idx]
    sst = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
    gain = (((Y - B) ** 2).sum(1) - ((Y - M) ** 2).sum(1)) / sst
    return float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))


def test_nitrogen_beats_both_baselines_at_five_steps():
    """The headline forecasting result."""
    r = horizon_fit("N_out", 5)
    assert r["rf"] == pytest.approx(0.516, abs=0.03)
    assert r["rf"] > r["roll"] and r["rf"] > r["persist"]


def test_baseline_skill_collapses_with_horizon():
    """Baselines decay to nothing at ten steps. The model does not."""
    r = horizon_fit("N_out", 10)
    assert r["roll"] == pytest.approx(0.0, abs=0.05)
    assert r["rf"] > 0.2


def test_nitrogen_advantage_at_five_steps_is_significant():
    """The interval on the gain over both baselines excludes zero."""
    r = horizon_fit("N_out", 5)
    assert advantage_ci(r["y"], r["pred"], r["roll_pred"])[0] > 0
    assert advantage_ci(r["y"], r["pred"], r["persist_pred"])[0] > 0


def test_cod_never_beats_a_rolling_mean():
    """Stated in the abstract, so it should fail loudly if it stops being true.

    The claim is about significance, not point estimates: at every horizon the
    interval on the gain over a rolling mean straddles zero.
    """
    for h in (1, 2, 3, 5, 7, 10):
        r = horizon_fit("COD_out", h)
        lo, hi = advantage_ci(r["y"], r["pred"], r["roll_pred"])
        assert lo <= 0 <= hi, f"COD pulled significantly ahead at horizon {h}"
