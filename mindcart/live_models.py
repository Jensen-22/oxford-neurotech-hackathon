"""Item-scoped 250 Hz Unicorn adapter for the two prototype model layers."""
from pathlib import Path
import numpy as np
from .video_liking_features import features_for_window, load_weights, WINDOW_SAMPLES

ROOT = Path(__file__).resolve().parents[1]
CHANNEL_INDEX = [0, 1, 3, 4, 5, 6, 7]  # Fz C3 C4 Pz PO7 Oz PO8


def window_features(raw):
    """[1250, 8] in µV, common headset reference -> 21 Cz-referenced features."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (WINDOW_SAMPLES, 8) or not np.isfinite(raw).all():
        return None
    # A flat/broken Cz invalidates every re-referenced channel.
    if np.std(raw[:, 2]) < 0.5:
        return None
    rereferenced = (raw[:, CHANNEL_INDEX] - raw[:, 2:3]).T
    features, _, _ = features_for_window(rereferenced)
    return features


def load_models():
    models = {}
    for name, filename in [('video_liking', 'video_liking_weights.npz'),
                           ('impulse', 'impulse_weights.npz'),
                           ('excitement', 'excitement_weights.npz')]:
        path = ROOT / 'models' / filename
        if path.exists():
            models[name] = load_weights(path)
    return models


def classify(features, model):
    return int(float(np.dot(features, model['weights']) + model['bias']) > 0)


class ItemModels:
    """Consume timestamped chunks and markers, allowing 1 s for marker arrival."""
    def __init__(self, models=None):
        self.models = load_models() if models is None else models
        self.samples = np.empty((0, 8)); self.times = np.empty(0); self.markers = []
        self.processed_until = None
        self.state = {'status': 'waiting', 'item_id': None, 'impulse': None,
                      'excitement': None, 'video_liking': None, 'available_models': sorted(self.models)}
        self.active = False; self.block = np.empty((0, 8)); self.good = []
        self.last_time = None; self.view_end = None; self.updated = False

    def event(self, marker, at=None):
        parts = marker.split('|'); kind = parts[0]
        if kind == 'img' and len(parts) >= 3:
            self.active = True; self.block = np.empty((0, 8)); self.good = []; self.last_time = None
            # The unchanged experiment displays each product for four seconds.
            self.view_end = at + 4.0 if at is not None else None
            self.state = {'status': 'collecting', 'item_id': parts[2], 'trial': parts[1],
                          'impulse': None, 'excitement': None, 'video_liking': None,
                          'usable_windows': 0, 'available_models': sorted(self.models)}
            self.updated = True
        elif kind in ('view_end', 'rate', 'fix', 'break', 'end') and self.active:
            self.active = False
            if self.state['status'] == 'collecting':
                self.state['status'] = 'insufficient_eeg'; self.updated = True
        if kind in ('rate', 'need', 'excitement') and len(parts) >= 4 and parts[1] == self.state.get('trial'):
            key = {'rate': 'reported_want', 'need': 'reported_need', 'excitement': 'reported_excitement'}[kind]
            try:
                value = int(parts[3])
            except ValueError:
                value = None
            self.state[key] = value if value in (1, 2, 3, 4, 5) else None
            want, need = self.state.get('reported_want'), self.state.get('reported_need')
            if want is not None and need is not None:
                if want >= 4 and need <= 2:
                    self.state['rated_impulse'] = 1
                elif want <= 2 or (want >= 4 and need >= 4):
                    self.state['rated_impulse'] = 0
                else:
                    self.state['rated_impulse'] = None
            self.updated = True

    def consume(self, x, t):
        if not self.active or self.state['status'] != 'collecting' or len(t) == 0:
            return
        reached_end = self.view_end is not None and t[-1] >= self.view_end
        if reached_end:
            stop = np.searchsorted(t, self.view_end, side='left')
            x, t = x[:stop], t[:stop]
        if not len(t):
            self.active = False; self.state['status'] = 'insufficient_eeg'; self.updated = True; return
        times = np.r_[self.last_time, t] if self.last_time is not None else t
        if len(times) > 1 and ((np.diff(times) <= 0).any() or (np.diff(times) > 0.012).any()):
            self.state['status'] = 'signal_interrupted'; self.active = False; self.updated = True; return
        self.last_time = t[-1]
        self.block = np.concatenate((self.block, x))
        while len(self.block) >= WINDOW_SAMPLES and len(self.good) < 3:
            f = window_features(self.block[:WINDOW_SAMPLES]); self.block = self.block[WINDOW_SAMPLES:]
            if f is not None:
                self.good.append(f)
            self.state['usable_windows'] = len(self.good); self.updated = True
        if len(self.good) == 3:
            f = np.mean(self.good, axis=0)
            for name in ('video_liking', 'impulse', 'excitement'):
                if name in self.models:
                    self.state[name] = classify(f, self.models[name])
            self.state['status'] = 'ready'; self.updated = True
        elif reached_end:
            self.active = False; self.state['status'] = 'insufficient_eeg'; self.updated = True

    def push(self, x, t, markers):
        x = np.asarray(x, dtype=float).reshape(-1, 8); t = np.asarray(t, dtype=float)
        if len(x) != len(t):
            raise ValueError('EEG sample and timestamp counts differ')
        for mt, marker in markers:
            if self.processed_until is not None and mt <= self.processed_until:
                # Late boundaries make item-to-sample alignment unreliable.
                kind = marker.split('|')[0]
                if kind == 'img' or (self.active and kind in ('view_end', 'fix', 'rate')):
                    self.active = False; self.state['status'] = 'marker_delayed'
                    for model_name in ('video_liking', 'impulse', 'excitement'):
                        self.state[model_name] = None
                    self.updated = True
            else:
                self.markers.append((mt, marker))
        self.markers.sort(key=lambda m: m[0])
        if len(t):
            self.samples = np.concatenate((self.samples, x)); self.times = np.r_[self.times, t]
        if not len(self.times):
            return
        cutoff = self.times[-1] - 1.0
        end = int(np.searchsorted(self.times, cutoff, side='right'))
        pos = 0
        while self.markers and self.markers[0][0] <= cutoff:
            mt, marker = self.markers.pop(0)
            idx = int(np.searchsorted(self.times, mt, side='left'))
            self.consume(self.samples[pos:idx], self.times[pos:idx]); self.event(marker, mt); pos = idx
        self.consume(self.samples[pos:end], self.times[pos:end])
        if end:
            self.processed_until = float(self.times[end-1])
            self.samples = self.samples[end:]; self.times = self.times[end:]
