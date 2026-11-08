"""Feature extraction and linear weights for the original video-liking model."""
import math
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch


# The original research-data channels and their approximate headset positions.
CHANNELS = [('E5', 'Fz_approx'), ('E42', 'C3_approx'), ('E93', 'C4_approx'),
            ('E62', 'Pz_approx'), ('E65', 'PO7_approx'), ('E75', 'Oz_approx'),
            ('E90', 'PO8_approx')]
BANDS = [('theta_4_8', 4, 8), ('alpha_8_13', 8, 13), ('beta_13_30', 13, 30)]
FS = 250
WINDOW_SAMPLES = FS*5
REQUIRED_WINDOWS = 3
NOTCH = tf2sos(*iirnotch(50, 30, fs=FS))
BANDPASS = butter(4, [1, 35], btype='bandpass', fs=FS, output='sos')

def features_for_window(window):
    """Turn one [7 channels, 1250 samples] window in microvolts into features."""
    if not np.isfinite(window).all():
        return None, 'non-finite samples', False
    frequencies, raw_power = welch(window, fs=FS, nperseg=500, axis=-1)
    line = raw_power[:, (frequencies >= 49) & (frequencies <= 51)].mean(axis=-1)
    neighbours = raw_power[:, ((frequencies >= 46) & (frequencies <= 48)) |
                              ((frequencies >= 52) & (frequencies <= 54))].mean(axis=-1)
    line_ratio = np.median(line/np.maximum(neighbours, 1e-30))
    # Reproduce the offline filtering and quality checks used during training.
    filtered = sosfiltfilt(NOTCH, window, axis=-1)
    filtered = sosfiltfilt(BANDPASS, filtered, axis=-1)[:, 125:-125]
    if np.ptp(filtered, axis=-1).max() > 200:
        return None, 'peak-to-peak signal over 200 uV', bool(line_ratio > 10)
    if np.std(filtered, axis=-1).min() < .5:
        return None, 'nearly flat channel', bool(line_ratio > 10)
    frequencies, power = welch(filtered, fs=FS, nperseg=500, nfft=1000, axis=-1)
    resolution = frequencies[1]-frequencies[0]
    features = []
    for index, _ in enumerate(CHANNELS):
        for _, low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            band_power = float(power[index, mask].sum()*resolution)
            features.append(math.log10(max(band_power, 1e-30)))
    return np.asarray(features), '', bool(line_ratio > 10)

def load_weights(path):
    """Read the model's 21 coefficients and bias."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'weights', 'bias'}:
            raise ValueError('Expected only weights and bias')
        coefficients = np.asarray(data['weights'], dtype=float)
        bias = np.asarray(data['bias'], dtype=float)
    if coefficients.shape != (21,) or bias.shape != () or not np.isfinite(coefficients).all() or not np.isfinite(bias):
        raise ValueError('Invalid model weights or bias')
    return {'weights': coefficients, 'bias': float(bias)}
