# Model integration

Open `dashboard/mindcart.html` for the shopping prototype, or `dashboard/index.html` for the live EEG dashboard. Start `python stream_unicorn.py` to record the Unicorn stream and send feature frames to the dashboards. `python stream_unicorn.py --fake` exercises the stream with synthetic EEG. Predictions and item status are written next to the recording as `*.predictions.jsonl`.

`models/video_liking_weights.npz` holds the existing linear model's coefficients and bias. The model is based on video-liking ratings, not shopping decisions. Its seven input channels are approximations of the original DENS channels. The stream adapter re-references the corresponding Unicorn signals to Cz; the headset/reference transfer has not been validated. The dashboard's existing interest index is calculated separately from band power, and demo-mode scores are simulated.

The existing experiment presents an item for four seconds. The saved model needs three usable five-second windows from the same item, so the current experiment reports insufficient EEG for an item instead of making a prediction from another item or a rating screen. No separately trained clothing impulse or excitement weights are included.

The experiment uses `stimuli/items.csv` and normalized clothing images in `stimuli/norm/`, which are not included. See `stimuli/SOURCE.txt` and `mindcart/README.md` for preparing images. Recordings and participant data are excluded from this commit.

To test the weights on an existing DENS sample in the original NPZ format:

```bash
python -m mindcart.predict_sample --eeg /path/to/example_raw_eeg.npz
```

`mindcart/train_layers.py` supports training separate impulse and excitement models once appropriately labelled recordings with at least 15 seconds of usable EEG per item are available. It holds participants out for testing and writes model weights and evaluation reports to `models/`. The unchanged four-second task does not provide the required viewing duration, and it does not collect excitement ratings.
