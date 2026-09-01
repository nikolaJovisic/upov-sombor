# UPOV Sombor

[![tests](https://github.com/nikolaJovisic/upov-sombor/actions/workflows/ci.yml/badge.svg)](https://github.com/nikolaJovisic/upov-sombor/actions/workflows/ci.yml)

Six years of operational monitoring from the municipal wastewater treatment plant in Sombor,
Serbia, and the code that analyses it. The plant serves about 18,000 population equivalents.
The record runs from January 2015 to July 2020.

## The paper

*Machine Learning for Effluent Forecasting at a Full-Scale Municipal Wastewater Treatment Plant:
Seasonal Performance Deterioration and the Decisive Role of Prediction Horizon.*
Clean Technologies, in revision.

Every number in the manuscript comes from `analysis/`. If a value appears in a table, in a figure
or in the abstract, one of these scripts printed it. Nothing is typed in by hand.

## What changed in 2026

The first submission filled the missing days before modelling. It reindexed the record to daily
frequency, interpolated in time, and smoothed with a fifty day rolling mean. Roughly half the
calendar days were never sampled. Models were therefore scored in large part against values the
pipeline had invented, and reconstructed values are smooth by construction. Over the same test
window that lifts the apparent coefficient of determination for effluent COD from 0.34 to 0.82.

The models also used different train and test splits from one another, so the comparison between
them was not valid.

Both problems are fixed. The analysis uses measured observations only, and 1,017 days survive
cleaning. Every model shares one chronological split. Training ends on 31 January 2019 and the
test period begins on 15 April 2019, a gap of 74 days, so no lagged feature of a test day can
reach back into data used for training.

The results are lower than the ones first submitted. They are also real.

## Layout

| Path | What it is |
| --- | --- |
| `analysis/canon.py` | The cleaned frame. This is the single definition, and every other script imports it. |
| `analysis/final.py` | Descriptive statistics, the seasonal result, all regression tables, exceedance classification. Writes `final_numbers.json`. |
| `analysis/horizon_final.py` | Forecast skill against horizon, for models and for both trivial baselines. Writes `horizon_numbers.json` and Figure D. |
| `analysis/robustness.py` | The same regression refitted on two earlier train and test periods. Section 3.4. |
| `analysis/imputation_effect.py` | What gap filling does to apparent accuracy. Section 3.7 and Figure 7. |
| `analysis/explain.py` | SHAP for the two models proposed for deployment. Writes `shap_numbers.json` and four figures. |
| `analysis/figures*.py` | Manuscript figures. They read the JSON files, so a figure cannot drift away from its table. |
| `analysis/exploratory/` | The working scripts, kept for the record. Nothing published depends on them. |
| `tests/` | The claims in the paper, asserted. Runs in CI on every push. |
| `dataset.py`, `model.py` | Data loading and the network definitions. Shared by both rounds of work. |
| `train.py`, `lin_reg_.py`, `prophet_.py` | The original 2023 analysis, which produced the first submission. Left as it was. |

## Running it

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then, in order:

```
python analysis/final.py             # a couple of minutes, trains the neural models on five seeds
python analysis/robustness.py
python analysis/imputation_effect.py
python analysis/horizon_final.py     # longer, six horizons and three targets
python analysis/figures.py           # needs final_numbers.json and imputation_numbers.json first
```

Scripts can be run from anywhere. They find the repository root themselves.

## Explaining the models

`analysis/explain.py` runs SHAP on the two models the paper recommends: the random forest that
flags effluent nitrogen above 15 mg/L, and the network that forecasts effluent BOD5 ten
measurements ahead. It produces a summary plot for each, showing both how much a feature matters
and which way it pushes, and two single day explanations.

The two single day plots are the point. One is an alarm the model raised. The other is a day when
nitrogen breached and the model stayed quiet, and it shows why: the largest single contribution is
the absence of a breach at the previous measurement, which pulls the prediction down.

shap does not install cleanly everywhere, so it lives in `requirements-explain.txt` rather than
`requirements.txt`. If pip fails on your machine, use the Dockerfile:

```
docker build -t upov-sombor .
docker run --rm -v "$PWD:/repo" upov-sombor python /repo/analysis/explain.py
```

## Tests

```
pytest tests/ -q
```

Ten checks, about twenty seconds, no neural training. They assert the things the paper actually
claims: the day count, that no gaps were filled, the summer nitrogen result and the fact that
suspended solids are unaffected, that nitrogen beats both baselines at five steps ahead, that
baseline skill collapses by ten steps while the model holds, and that COD never pulls
significantly ahead of a rolling mean.

Data assertions are exact. Model assertions carry tolerances, because point estimates move a
little between scikit-learn releases. That is also why `requirements.txt` is pinned.

The full pipeline, neural models included, runs as a manual job from the Actions tab and uploads
the JSON files and figures as artifacts.

## The data

`data.csv` holds one row per sampling day, with influent and effluent concentrations for chemical
oxygen demand, biochemical oxygen demand, nitrogen, phosphorus and suspended solids, plus
temperature, pH and hydraulic flow. Effluent columns end in `_out` and influent columns in `_in`.

Cleaning drops duplicate records, rows with missing influent or effluent values, implausible flows
and zero readings, and averages any remaining repeat measurements on the same date. The cascade is
printed by `final.py` at every step.
