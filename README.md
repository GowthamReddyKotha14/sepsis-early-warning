# Sepsis Early Warning

**Can an AI warn ICU staff about sepsis hours earlier, without burying nurses in false alarms?**

Sepsis is the body's extreme response to an infection. Every hour that treatment is delayed raises the risk of death, so hospitals want earlier warnings. But warning systems that fire too often get ignored ("alarm fatigue"). This project builds and honestly evaluates an early-warning model on real ICU data, and replays patients hour by hour as a live ward.

> **Research and education project. Not a medical device. Not for clinical use.**

## Data

[PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/): 40,336 de-identified ICU patients from two hospital systems, one row per patient-hour, up to 40 variables (8 vital signs, 26 lab values, 6 demographic/admin fields). Sepsis onset follows the Sepsis-3 definition; `SepsisLabel` turns 1 six hours before onset, so a model that fires on the label is warning six hours early.

Reyna MA, et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Crit Care Med. 2020;48(2):210-217.

## Evaluation design

- **Split by patient, never by hour.** Hours from one patient are highly correlated; mixing them across train and test leaks information.
- **Hospital A trains, hospital B tests.** Hospital A is split 80/20 into train and validation (stratified by sepsis). Hospital B is held out entirely, as an external test of whether the model transfers to a hospital it never saw.
- **Scored the way the challenge scored it**, with the official utility metric (reimplemented and checked against PhysioNet's code), plus lead time and alarms per nurse shift.

## Results (hospital B, never used for training or tuning)

| | Score (official utility; 1 = perfect, 0 = never alarm) | Alerts per bed per 12-h shift | Alerts on patients who became septic | Septic patients warned before onset (of 1,142) | Hourly AUROC |
|---|---|---|---|---|---|
| SIRS ≥ 2 (rule) | −0.014 | 0.67 | 10% | 966 | n/a |
| qSOFA, adapted (rule) | +0.014 | 0.19 | 14% | 429 | n/a |
| Model, all features | +0.201 | 0.13 | 20% | 590 | 0.717 |
| **Model, physiology only** | **+0.134** | **0.10** | **24%** | **520** | **0.713** |

Alarm levels were chosen on hospital A validation patients under a budget of 0.30 alerts per bed per shift (qSOFA's load there). Raw numbers: `results/`.

### The shortcut investigation

Explaining every hospital B alert showed the all-features model's top reason was **hours in ICU** for 64% of real-sepsis alerts and **time since hospital admission** for 44% of false alarms: it was largely reading the clock. Some of that is real risk (long stays bring infections), and some is how the dataset was assembled.

1. Removing the two time features cut the score from +0.201 to +0.109, and the new top reason became the **running count of labs drawn** (one alert: "335 labs drawn"). That count grows with length of stay, so it was a clock too.
2. Replacing it with **labs drawn in the last 24 hours** raised the score to **+0.134** with the fewest alarms of any approach, and the top reasons became physiology: lactate, temperature, bilirubin, fibrinogen, shock index.

So the physiology-only model is the one I'd trust: it scores lower than the all-features model, but for reasons a clinician would recognize.

### Honest limits

- Hourly AUROC is about 0.71 at the new hospital; the model ranks risk modestly well, not perfectly.
- Median "first warning before onset" is about 26 hours, but that is inflated: many patients are flagged long before sepsis develops, which the official score penalizes. Don't read it as "warns 26 hours early".
- The alarm level tuned at hospital A gave fewer alarms at hospital B (0.10 vs a 0.30 budget), because B has less sepsis (5.7% vs 8.8%). An alarm setting needs re-checking at each hospital.
- Not comparable to the challenge leaderboard (top official score 0.360), which scored hidden data from three hospitals including ones teams trained on.

## The model

| Step | What it does | Why |
|---|---|---|
| Features (`sepsis/features.py`) | Latest value of each vital and lab, hours since each was measured, 6-hour means and changes of vitals, shock index, labs drawn so far | Uses only what was known at that hour. A test changes a patient's future and checks earlier features don't move |
| Model (`sepsis/model.py`) | Gradient-boosted trees, trained on hospital A's training patients only | Handles missing labs natively; trains in minutes on a laptop |
| Alarm level (`sepsis/threshold.py`) | Chosen on validation patients: best challenge utility within an alarm budget (default 0.30 alerts per bed per 12-hour shift, qSOFA's load on the same patients) | Compares the model and the rule at the same number of alarms nurses already get |
| Final test | Hospital B, scored once | Would it work at a hospital it never saw? |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make data        # download PhysioNet 2019 (about 41 MB)
make audit       # cohort summary -> results/audit.json
make baseline    # SIRS / qSOFA scores -> results/baseline.json
make train       # features + model on hospital A -> models/sepsis_hgb.joblib
make evaluate    # alarm level on validation, then hospital B once -> results/model.json
make test        # tests use small synthetic patients, never the real data
```

## Status

Built step by step (see the commit history). Done: data pipeline, official utility metric, patient-level split with an external hospital, clinical baselines, time-respecting features, the model, alarm-level selection, explanations for every alert, and the shortcut investigation above. Next: a live ward replay and showcase site.

[![ci](https://github.com/GowthamReddyKotha14/sepsis-early-warning/actions/workflows/ci.yml/badge.svg)](https://github.com/GowthamReddyKotha14/sepsis-early-warning/actions/workflows/ci.yml)
