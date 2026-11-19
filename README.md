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

Work in progress, built step by step (see the commit history). Done: data pipeline, official utility metric, patient-level split with an external hospital, clinical baselines, time-respecting features, the model, and alarm-level selection. Next: explanations for each alert and a live ward replay.

[![ci](https://github.com/GowthamReddyKotha14/sepsis-early-warning/actions/workflows/ci.yml/badge.svg)](https://github.com/GowthamReddyKotha14/sepsis-early-warning/actions/workflows/ci.yml)
