# MindCart: run sheet

**Question:** does EEG improve clothing-preference prediction over product data alone?

## TONIGHT: the exact run order (Mac + Unicorn over Bluetooth)

```bash
# 0. once
pip3 install numpy scipy pandas pillow scikit-learn matplotlib pylsl websockets pyserial psychopy
# 1. outfits: from the PowerPoint board (or drop 60-80 photos in stimuli/raw/ yourself)
python3 -m mindcart.pptx_to_stimuli ~/Downloads/Grace_Fashion_Board.pptx --out stimuli/raw
python3 -m mindcart.prep_images stimuli/raw stimuli/items.csv --out stimuli/norm   # auto-creates items.csv
# 2. headset check (eyes open, eyes closed, blinks) -> must say READY
python3 stream_unicorn.py --check
# 3. ERP check, ~4 min.  Terminal 1:
python3 stream_unicorn.py --name OB01.csv
#                         Terminal 2:
python3 mindcart/experiment.py --pid OB01 --oddball
python3 -m mindcart.erp recordings/OB01.csv --oddball      # PASS = markers + ERPs work
# 4. each participant, ~12 min.  Terminal 1:
python3 stream_unicorn.py --name P01.csv
#                         Terminal 2:
python3 mindcart/experiment.py --pid P01 --items stimuli/items.csv --img-dir stimuli/norm
# 5. analysis (all participants at once)
python3 -m mindcart.features recordings/P01.csv recordings/P02.csv
python3 -m mindcart.erp      recordings/P01.csv recordings/P02.csv
python3 -m mindcart.models   recordings/P01.csv recordings/P02.csv --items stimuli/items.csv
```

- `stream_unicorn.py` writes EEG **and** the experiment's markers into ONE file, so there's no separate recorder.
- Results land in `recordings/P01/` (trials.csv, erp_want.png) and `recordings/results.png`.
- Ratings: **WANT** (1–5) then **NEED** (1–5). Main analysis = predict WANT from the brain.
  UnSway rule = brain-predicted WANT + low stated NEED → nudge. `--label impulse` (want ≥ 4 & need ≤ 2) is
  exploratory, because it leaves only about 10 positive trials per person.
- Go/no-go at about 4 AM: oddball PASS and at least 2 participants recorded → use this. Otherwise fall back to the dashboard.

## 0. Setup (once)

```bash
pip install -r requirements.txt psychopy scikit-learn pandas pillow pylsl
```

Put the images in `stimuli/raw/`. Fill in `stimuli/items.csv` (template: `mindcart/items_template.csv`):

```
item_id,file,category,colour,price,style
I001,blue_hoodie.jpg,hoodie,blue,45,casual
```

`item_id` must not contain `|`. Then normalise the images (same size, same grey background) and compute brightness and colourfulness for each:

```bash
python -m mindcart.prep_images stimuli/raw stimuli/items.csv --out stimuli/norm
```

**Use 60–80 items.** Mix the categories and colours, and avoid famous brands and logos.

## 1. Before each participant (questionnaire, *before* EEG)

Create `sessions/P01/prefs.csv` (template: `mindcart/prefs_template.csv`). Rate each category and each colour from 1 to 5. This is Model A's "prior preferences". It must be collected **before** the session so it can't leak the ratings you're trying to predict.

## 2. Record

1. Headset on. In Unicorn Suite, check the signal until every channel is quiet. Our first recordings were 50–28,000 µV and we need about 10–40 µV.
2. Start **Unicorn LSL** streaming.
3. Terminal 1: `python -m mindcart.record_session --pid P01`
4. Terminal 2: `python mindcart/experiment.py --pid P01 --items stimuli/items.csv --img-dir stimuli/norm`
5. The recorder stops by itself when the experiment ends.

**Sync check first:** run `--pid TEST --n 5 --windowed`, then step 3 below on it. You should see 5 trials.

A session lasts about 8 s per trial, so 80 items take about 11 minutes, with a break halfway.

## 2b. ERP check: P300 oddball (do this once, ~4 min)

Proves the markers, timing and headset record real brain responses before you spend time on the product task.

```bash
python -m mindcart.record_session --pid OB01            # terminal 1
python mindcart/experiment.py --pid OB01 --oddball      # terminal 2: count the orange squares
python -m mindcart.erp sessions/OB01 --oddball          # -> sessions/OB01/oddball_p300.png + PASS/FAIL
```

PASS means the rare orange squares give a P300 at Pz (280–480 ms) that is clearly larger than for the blue circles.

## 3. Features → models

```bash
python -m mindcart.features sessions/P01 sessions/P02
python -m mindcart.models   sessions/P01 sessions/P02 --items stimuli/items.csv
```

This gives `sessions/results.csv` and `sessions/results.png`, which is the slide for the pitch.

- **Labels:** like = 4–5 and dislike = 1–2; 3s are dropped. Use `--like median` if too few trials remain.
- **EEG features (8, hypothesis-driven):** Fz theta and Fz beta, posterior and central alpha, C4–C3 and PO8–PO7 alpha asymmetry, β/α, and posterior theta. Each is a log change from the fixation baseline, measured 0.5–4 s after image onset. Use `--full` for all 24. The Unicorn has no F3/F4, so the classic frontal alpha asymmetry isn't available.
- **Evaluation:** within each participant, 5-fold CV repeated 10 times, giving balanced accuracy, F1 and a majority baseline. Then a 500-shuffle **permutation test** for the p-value and the 95% chance line. **C − A** is the headline number: what EEG adds on top of product data.
- **Artefacts:** trials with more than 200 µV peak-to-peak in the analysis window are dropped (`--ptp`).
- **ERP features (P300 + LPP):** filtered 0.1–30 Hz (slow waves survive), epochs −200 to 1000 ms around image onset, baseline −200 to 0 ms, epochs over ±100 µV at Fz/Cz/Pz rejected.
  - `f_p300_pz`, `f_p300_cz`: mean 250–500 ms. Attention, "this caught my eye".
  - `f_lpp_pz`, `f_lpp_czpz`: mean 500–1000 ms. Sustained motivational significance (interest or liking).
  - Rejected epochs get NaN and are imputed inside each CV training fold.
- **ERP plot:** `python -m mindcart.erp sessions/P01` gives `sessions/P01/erp.png` (liked vs not liked at Pz and Cz).

## Test without hardware

```bash
python -m mindcart.simulate --out sim --effect 2.0    # strong EEG effect
python -m mindcart.features sim/P01 sim/P02 && python -m mindcart.models sim/P01 sim/P02 --items sim/items.csv
```

Checked on simulated data with 70 items and 2 participants:

| Simulated EEG effect | A: product | B: EEG | C: combined | C − A |
|---|---|---|---|---|
| none | 0.60–0.63 | 0.48 (p ≈ 0.6–0.7, chance ✔) | 0.56–0.58 | −0.04 |
| weak (0.6) | 0.60–0.63 | 0.50–0.59 (n.s.) | 0.61–0.64 | +0.01 |
| strong (2.0) | 0.60–0.63 | **0.76–0.82 (p ≤ 0.01)** | **0.75–0.83** | **+0.15–0.20** |
| medium (1.0) + P300/LPP | 0.60–0.62 | 0.64–0.79 (p ≤ 0.025) | 0.64–0.84 | +0.05–0.22 |

So when there's no real effect, the permutation test correctly says chance. A weak effect isn't detectable with about 46 usable trials, so more trials per person matter more than more people.

## Honest limitations (for the pitch)

- With 2–4 participants and about 50 usable trials each, the results are exploratory.
- A rating of interest is not purchase intention.
- Brightness and colour affect occipital alpha. Model A includes them, so C − A is the fair test.
- No claims about reducing impulse buying without a follow-up study.
