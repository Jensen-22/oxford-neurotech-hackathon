# MindCart: EEG-powered shopping research prototype

MindCart is a prototype for exploring whether EEG can add useful information while someone looks at clothing. The intended application is a shopping assistant that encourages people to pause before an unplanned purchase or before buying an item they do not feel strongly about.

The project connects an eight-channel **g.tec Unicorn Hybrid Black** headset to an experiment and browser dashboards. It records EEG while clothing is shown, collects the participant's want and need ratings, calculates signal features, and includes code for applying a trained classifier.

## What works today

- Live or simulated EEG can be recorded and displayed in the dashboards.
- The experiment shows products and records want/need answers with timestamps.
- The included model can classify suitable DENS dataset samples as **lower** or **higher video liking**.
- The dashboard shows model status and, when present, model results alongside the separate band-power-based interest index.
- A rule can describe a participant's *reported* answers as high want and low need after the questions are answered.

**Current limitation:** the included weights were trained on video-liking ratings, not clothing purchases or impulse-buying outcomes. The existing experiment shows each item for four seconds, whereas the model needs at least 15 seconds of usable EEG for the same item. The current live adapter therefore reports `insufficient_eeg` for those trials. The displayed interest index is not the output of the trained model, and the shopping dashboard does not currently make its decision using the model.

## End-to-end flow

```text
                 Clothing experiment                 Unicorn headset
            fixation → product → ratings            8 EEG channels, 250 Hz
                       │                                     │
              item and answer markers                        │
                       └──────────────┬──────────────────────┘
                                      ▼
                         stream_unicorn.py
              saves timestamped EEG + markers; sends live frames
                                      │
                   ┌──────────────────┴───────────────────┐
                   ▼                                      ▼
        Live dashboard features                    Model adapter
        2-second windows                           item-specific EEG
        band power + signal quality                → preprocessing + quality checks
        → interest index                           → 21 features
        (separate from the classifier)             → saved weights + bias
                                                   → higher/lower liking, if available
```

Afterward, the want and need answers can be used as **labels for future training**. They are not inputs to the model while predicting the EEG from the item viewing period.

## 1. Record the headset and show products

The headset stream is expected to have these channels, in order:

```text
Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
```

`stream_unicorn.py` supports Bluetooth serial, Lab Streaming Layer (LSL), UnicornPy, and synthetic EEG. It records timestamped EEG and experiment markers in one CSV file. The stream is expected to be sampled at **250 Hz** with EEG in **microvolts (µV)**.

The PsychoPy experiment in `mindcart/experiment.py` performs each trial as follows:

1. Show a fixation cross for one second.
2. Show a clothing image for four seconds and send an image-onset marker containing the trial and item ID.
3. Ask the participant to rate how much they **want** the item, then how much they **need** it, each from 1 to 5.
4. Send markers containing those answers and save a behavioural CSV as a backup.

The markers align the item and ratings with the EEG. The model adapter keeps EEG within the item viewing period; it does not take samples from the later rating screens or the next item to fill a window.

## 2. Process EEG for the live dashboard

The live dashboard feature stream and the model feature path perform different calculations.

For the live dashboard, `stream_unicorn.py`:

1. Applies a stateful **1–40 Hz band-pass filter** and a **50 Hz notch filter** to reduce slow drift, high-frequency activity and mains interference.
2. Uses a two-second window and updates the feature frame approximately once per second.
3. Checks channels for flat, noisy or large-amplitude signals.
4. Calculates spectral power in delta, theta, alpha and beta bands using Welch's method.
5. Sends the features and quality information to the dashboard over a WebSocket.

MindCart's interest index is calculated from band-power changes; it is **not** the saved classifier's output. The built-in dashboard demo uses simulated scores and does not run the model.

## 3. Prepare model input

The included model uses seven positions corresponding approximately to its original research-dataset channels, in this order:

```text
Fz, C3, C4, Pz, PO7, Oz, PO8
```

The Unicorn adapter uses Cz to re-reference these seven channels, assuming the headset values share a suitable common reference. The original dataset electrodes only approximate these Unicorn positions; transfer to Unicorn hardware has not been validated.

For each model prediction, `mindcart/live_models.py` and `mindcart/video_liking_features.py`:

1. Collect **five-second** windows: 1,250 samples per channel at 250 Hz.
2. Apply a 50 Hz notch filter and a **1–35 Hz** band-pass filter to each completed window. This is the model's preprocessing, separate from the dashboard filter.
3. Remove the first and last half-second of the filtered window and reject windows containing non-finite values, excessive amplitude or an almost-flat channel.
4. Calculate log band power using Welch's method in **theta (4–8 Hz)**, **alpha (8–13 Hz)** and **beta (13–30 Hz)** for each of the seven channels.
5. Average the first three passing windows, giving **21 features**: seven channels × three frequency bands. The order is channel-first, for example Fz theta, Fz alpha, Fz beta, C3 theta, and so on.

The three windows require at least 15 seconds of EEG. Rejected windows can increase the amount of time required. The code does not substitute data from a different item when there is too little usable EEG.

## 4. Apply the trained model

The saved file is `models/video_liking_weights.npz`. It contains only:

- `weights`: 21 coefficients, one for each feature.
- `bias`: one intercept value.

The original model was trained as a **linear logistic classifier**, using scikit-learn's `SGDClassifier` with logistic loss for ten training passes. Standardisation fitted on the training data has already been folded into the exported weights and bias.

At inference, the code calculates:

```text
score = (feature 1 × weight 1) + ... + (feature 21 × weight 21) + bias
```

A score greater than zero maps to **higher liking**; otherwise it maps to **lower liking**. The sample-prediction script outputs the class, not a calibrated confidence score.

The weights come from the [DENS dataset, OpenNeuro ds003751, version 1.0.6](https://openneuro.org/datasets/ds003751/versions/1.0.6). The training task used ratings of videos: ratings of 1–2 were treated as low liking, ratings of 4–5 as high liking, and middle ratings were excluded. Those labels do not establish whether a person would make an impulse purchase. An earlier prototype renamed the two classes as impulse and non-impulse; renaming them did not change what the model learned.

### Previous held-out test

In a previous local evaluation of these weights, the classifier was trained on **93 trials from 18 participants** and tested on **23 trials from five different participants**. On a short-window version of that held-out test, it matched **16 of 23 video-liking labels (69.6% accuracy)**. Always choosing the low-liking class would have matched **14 of 23 (60.9%)**.

This is an exploratory video-liking result, not an impulse-buying accuracy. The dataset had been explored before this split was created, some videos appeared in both partitions, and the short-window model has not been validated on Unicorn clothing recordings. The held-out EEG samples and the original evaluation report are not included in this repository.

## 5. Send and save the result

`stream_unicorn.py` sends feature frames to `ws://localhost:8765` and raw EEG to `ws://localhost:8766`. The feature frame contains a `model` object with the current item, status, available models and any results. Possible statuses include `waiting`, `collecting`, `ready`, `insufficient_eeg`, `signal_interrupted` and `marker_delayed`.

The model panel on the dashboards reads this object. A result appears only when the backend provides one; demo-mode values are identified as simulated. A want/need result after the questionnaire is based on those answers, not on the EEG classifier.

Files produced by a recording named `P01.csv` include:

```text
recordings/P01.csv                 Timestamped EEG and markers
recordings/P01.predictions.jsonl   Model status and available results, one JSON object per line
sessions/P01/behaviour.csv         Trial and want/need answers
```

## Running the project

### Install

From a terminal, clone the repository, create a Python environment and install the dependencies:

```bash
git clone https://github.com/Jensen-22/oxford-neurotech-hackathon.git
cd oxford-neurotech-hackathon
python3 -m venv .venv
source .venv/bin/activate                 # macOS / Linux
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. PsychoPy is required for the product experiment; UnicornPy is only needed for that particular headset connection method.

### View the dashboards

Open `dashboard/mindcart.html` for the shopping prototype. `dashboard/index.html` shows live EEG features, and `dashboard/pipeline.html` and `dashboard/brainwaves.html` provide additional EEG views.

On macOS, for example:

```bash
open dashboard/mindcart.html
```

Select **Run demo** to explore the dashboard with simulated data. To send synthetic EEG through the backend instead, start:

```bash
python stream_unicorn.py --fake
```

Then select **Connect headset** in the dashboard. Synthetic EEG tests the connection and display; it is not a participant prediction. Stop the stream with `Ctrl+C`.

### Run a headset session

Put clothing photos in `stimuli/raw/`, then generate the item list and normalised images:

```bash
python -m mindcart.prep_images stimuli/raw stimuli/items.csv --out stimuli/norm
```

The photos and generated `stimuli/norm/` folder are not supplied in this repository. See `stimuli/SOURCE.txt` and `mindcart/README.md` for the original image source and task details.

Start the headset stream in one terminal, using the connection method appropriate for your device:

```bash
python stream_unicorn.py --name P01.csv
# Or, when using the Unicorn LSL application:
python stream_unicorn.py --source lsl --name P01.csv
```

In a second terminal with the environment activated, start a short experiment:

```bash
python mindcart/experiment.py --pid P01 --n 5 --windowed
```

Connect the dashboard to the stream, follow the on-screen task and answer the want/need questions. The current four-second trials will produce `insufficient_eeg` rather than a model prediction, as described above.

### Test the existing model on a DENS sample

The sample must already be in the specific NPZ format expected by the script; a raw dataset file is not directly accepted. It needs `windows_uv` shaped `[number of windows, 7, 1250]`, `sampling_rate_hz` of 250, DENS `channel_names` in the expected order, `units` set to `uV`, and `source_reference` set to `E129/Cz`. The first three passing windows are used.

```bash
python -m mindcart.predict_sample --eeg /path/to/example_raw_eeg.npz
```

The script prints the video-liking class and saves it to `recordings/sample_model_prediction.json`. The dataset sample is not bundled in this repository.

## Training clothing-specific models

`mindcart/train_layers.py` provides a starting point for separate clothing-related models. It expects recordings containing suitable item markers, labels and at least three usable five-second windows **within the same item**. It uses one file per participant and performs a participant-grouped train/test split; the scaler is fitted on the training participants only. It fits a separate class-balanced logistic regression for each target and saves weights and evaluation reports to `models/`.

For longer recordings, the training loader recognizes an explicit end marker and optional excitement rating in addition to the existing markers:

```text
img|<trial>|<item_id>
view_end|<trial>|<item_id>
rate|<trial>|<item_id>|<want>|<response_time>
need|<trial>|<item_id>|<need>|<response_time>
excitement|<trial>|<item_id>|<excitement>|<response_time>
```

The unchanged experiment does not send `view_end` or `excitement` markers. Without `view_end`, the training loader assumes the original four-second viewing period.

Its current example label rules are:

| Target | Positive class | Negative class |
|---|---|---|
| Impulse-related proxy | Want ≥ 4 and need ≤ 2 | Want ≤ 2, or want ≥ 4 and need ≥ 4 |
| Excitement | Excitement ≥ 4 | Excitement ≤ 2 |

Other or missing ratings are excluded where the target cannot be assigned. High want and low need is a **self-report proxy**, not a measured purchase outcome. The current task collects want and need, but does **not** ask for an excitement rating. No separately trained clothing impulse or excitement weights are supplied.

For recordings that meet the requirements, with one distinct participant per file:

```bash
python -m mindcart.train_layers recordings/P01.csv recordings/P02.csv recordings/P03.csv
```

At least three participants, both classes and a viable split are required. Reports include held-out accuracy, balanced accuracy and a confusion matrix. Restart the stream to load newly saved weights. The test split groups by recording path, so do not provide multiple files from the same participant as if they were different people; the split also does not guarantee unseen products.

**Before this can produce live clothing-model predictions**, both the task and live adapter need to support a long enough item viewing period and matching end markers. The adapter currently stops each item after four seconds, so increasing only the display time will not be sufficient. Collect excitement ratings separately if training that target, and evaluate the resulting models on held-out people and products.

## Project layout

| Path | Purpose |
|---|---|
| `stream_unicorn.py` | Headset input, recording, live features and WebSocket streams |
| `dashboard/` | MindCart, EEG and pipeline browser views |
| `mindcart/experiment.py` | Timed product-viewing task and want/need questions |
| `mindcart/live_models.py` | Item timing, model input preparation and classification |
| `mindcart/video_liking_features.py` | Filtering, quality checks, band-power features and weight loading |
| `mindcart/predict_sample.py` | Prediction on a compatible DENS sample |
| `mindcart/train_layers.py` | Training and held-out evaluation for future clothing models |
| `models/video_liking_weights.npz` | Existing video-liking coefficients and bias |
| `eeg_pipeline/` | Reusable EEG processing and signal-quality utilities |
| `tests/` | Pipeline, ERP and model-adapter tests |

Run the tests with:

```bash
python -m pytest -q
```

See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) and [mindcart/README.md](mindcart/README.md) for additional implementation and experiment details.
