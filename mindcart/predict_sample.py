"""Run the video-liking weights on a known-format DENS EEG sample.

python -m mindcart.predict_sample --eeg /path/to/example_raw_eeg.npz
"""
import argparse
import json
from pathlib import Path
import numpy as np
from mindcart.video_liking_features import CHANNELS, FS, WINDOW_SAMPLES, features_for_window, load_weights
from mindcart.live_models import classify, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eeg', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models' / 'video_liking_weights.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'recordings' / 'sample_model_prediction.json')
    args = parser.parse_args()
    with np.load(args.eeg, allow_pickle=False) as sample:
        required = {'windows_uv', 'sampling_rate_hz', 'channel_names', 'units', 'source_reference'}
        if not required.issubset(sample.files):
            raise ValueError('The EEG sample lacks required format metadata')
        if float(sample['sampling_rate_hz'].item()) != FS or str(sample['units'].item()) != 'uV':
            raise ValueError('Expected EEG at 250 Hz in microvolts')
        if list(sample['channel_names'].astype(str)) != [name for name, _ in CHANNELS] or str(sample['source_reference'].item()) != 'E129/Cz':
            raise ValueError('Expected the DENS channel order and E129/Cz reference')
        windows = np.asarray(sample['windows_uv'], dtype=float)
    if windows.ndim != 3 or windows.shape[1:] != (7, WINDOW_SAMPLES):
        raise ValueError('Expected EEG windows shaped [n, 7, 1250]')
    good = []
    for window in windows:
        features, _, _ = features_for_window(window)
        if features is not None:
            good.append(features)
        if len(good) == 3:
            break
    if len(good) < 3:
        raise ValueError('Three usable five-second windows are required')
    prediction = classify(np.mean(good, axis=0), load_weights(args.weights))
    result = {'model': 'video_liking', 'prediction': 'higher_liking' if prediction else 'lower_liking',
              'usable_windows': len(good), 'source': str(args.eeg)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
