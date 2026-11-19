# Sepsis Early Warning

**Can an AI warn ICU staff about sepsis hours earlier, without burying nurses in false alarms?**

Sepsis is the body's extreme response to an infection. Every hour that treatment is delayed raises the risk of death, so hospitals want earlier warnings. But warning systems that fire too often get ignored ("alarm fatigue"). This project builds and honestly evaluates an early-warning model on real ICU data, and replays patients hour by hour as a live ward.

> **Research and education project. Not a medical device. Not for clinical use.**

## Data

[PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/): 40,336 de-identified ICU patients from two hospital systems, one row per patient-hour, up to 40 variables (8 vital signs, 26 lab values, 6 demographic/admin fields). Sepsis onset follows the Sepsis-3 definition; `SepsisLabel` turns 1 six hours before onset, so a model that fires on the label is warning six hours early.

Reyna MA, et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Crit Care Med. 2020;48(2):210-217.

## Status

Work in progress. Built step by step; see the commit history.
