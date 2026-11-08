# UnSway: analysis plan (fixed BEFORE recording, 7–8 Nov 2026)

Written before any participant data exists, so results can't steer the analysis.
Everything below is reported, whatever it shows.

## Question
Does EEG during the first seconds of viewing an outfit predict **WANT**, beyond what the image and
product information already predict?

## Design
- Unicorn Hybrid Black, 8 ch (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8), 250 Hz.
- 72 outfit photos (CC0 clothing dataset, 9 categories), normalised to 600×600 on grey.
- Trial: fixation 1 s → outfit 4 s → WANT 1–5 → NEED 1–5. Order randomised per participant.
- Target: 2–3 participants, each analysed separately (within-participant).

## Validation gate (oddball, before any outfit sessions)
200 stimuli (20% targets), participant counts targets.
**PASS** = mean amplitude at Pz 250–500 ms, target − standard > 0, one-sided trial-level permutation
test (5000 shuffles) p < .05. FAIL → fix setup before recording participants.

## Preprocessing (fixed)
- Band power: 1–40 Hz + 50 Hz notch (zero-phase) on the continuous recording; Welch log power in θ 4–8,
  α 8–13, β 13–30 Hz; image window 0.5–4.0 s; baseline = the participant's mean fixation power (−0.9 to −0.1 s)
  over clean trials. Reject a trial if peak-to-peak > 200 µV in the image window.
- ERP: 0.1–30 Hz + notch on the continuous recording, then epochs −0.2 to 1.0 s, baseline −0.2 to 0 s;
  reject the epoch if |amplitude| > 100 µV at Fz/Cz/Pz.
- **Rejection rates are reported per participant.**

## Features (fixed, 12, chosen by hypothesis, not by looking at data)
Band power (8): Fz θ, Fz β, posterior α (Pz/PO7/Oz/PO8), central α (C3/Cz/C4), α asymmetry C4−C3 and PO8−PO7,
β−α, posterior θ. ERP (4): P300 Pz, P300 Cz (250–500 ms), LPP Pz, LPP Cz+Pz average (500–1000 ms).
The 24-feature "full" set is exploratory only.
Exploratory breakdown (added 8 Nov, before recording): B split into band power only (8) vs ERP only (4),
reported as `Bx_band` / `Bx_erp`. Not used for the primary claim.

## Labels
- **Primary:** WANT 4–5 = 1, WANT 1–2 = 0, WANT 3 excluded.
- **Exploratory:** impulse = WANT ≥ 4 & NEED ≤ 2 (1) vs WANT ≤ 2 or (WANT ≥ 4 & NEED ≥ 4) (0).

## Models (fixed)
- A = image/product info (category, luminance, contrast, colourfulness).
  B = EEG (12 features). C = A + B.
- Logistic regression, L2, C = 0.5 (fixed, not tuned), class-balanced; mean imputation + scaling fitted inside folds.
- Stratified 5-fold CV × 10 repeats; balanced accuracy; F1; majority baseline.
- Permutation test: 500 label shuffles, full pipeline refit on each, per participant.
- Sensitivity: train on first half / test on second half (and reverse); should agree with CV.

## What we will claim
- **Primary claim = C beats A** (EEG adds information beyond the image). "B beats chance" alone is not
  enough, because visual channels and ERPs also respond to the image itself.
- Report all participants × {A, B, C} × {want (primary), impulse (exploratory)}, including nulls.
- With 2–3 participants the results are **exploratory / feasibility**, not population claims.

## Live demo
The live model is trained on the participant's 72 rated outfits; the demo shows **11 unseen outfits**
(`stimuli/demo/`), never outfits from the training set.
