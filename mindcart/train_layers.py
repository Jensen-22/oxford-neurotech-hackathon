"""Train separate Unicorn clothing models with a participant-held-out test split.

python -m mindcart.train_layers recordings/P01.csv recordings/P02.csv recordings/P03.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from mindcart.erp import load, parse_trials
from mindcart.live_models import window_features


def get_rows(paths):
    rows = []
    for path in paths:
        t, x, markers = load(path)
        if len(t) < 2 or not 245 <= 1.0 / np.median(np.diff(t)) <= 255:
            raise ValueError(f'{path}: expected 250 Hz EEG timestamps')
        trials = parse_trials(markers)
        # Optional explicit markers are supported for recordings from longer tasks.
        view_ends, excitement = {}, {}
        for timestamp, marker in zip(markers['lsl_time'], markers['marker']):
            parts = str(marker).split('|')
            if len(parts) >= 3 and parts[0] == 'view_end':
                view_ends[int(parts[1])] = float(timestamp)
            elif len(parts) >= 4 and parts[0] == 'excitement':
                try: excitement[int(parts[1])] = float(parts[3])
                except ValueError: pass
        for _, tr in trials.iterrows():
            if int(tr['trial']) in excitement:
                tr = tr.copy(); tr['excitement'] = excitement[int(tr['trial'])]
            onset = float(tr['onset'])
            # Never extend a window into questions, the next item, or a recording break.
            end = view_ends.get(int(tr['trial']), onset + 4.0)
            i, j = np.searchsorted(t, [onset, end])
            tx, xx = t[i:j], x[i:j]
            good = []
            for start in range(0, len(xx) - 1250 + 1, 1250):
                ts = tx[start:start+1250]
                if np.any(np.diff(ts) <= 0) or np.any(np.diff(ts) > .012):
                    break
                f = window_features(xx[start:start+1250])
                if f is not None:
                    good.append(f)
                if len(good) == 3:
                    break
            if len(good) < 3:
                continue
            rows.append((str(Path(path).resolve()), tr, np.mean(good, axis=0)))
    return rows


def label(tr, target):
    want, need, excitement = (tr.get(k, np.nan) for k in ('rating', 'need', 'excitement'))
    if target == 'impulse':
        if want >= 4 and need <= 2:
            return 1
        if want <= 2 or (want >= 4 and need >= 4):
            return 0
    else:
        if excitement >= 4:
            return 1
        if excitement <= 2:
            return 0
    return None


def train(rows, target, out):
    selected = [(group, label(tr, target), x) for group, tr, x in rows if label(tr, target) is not None]
    if not selected:
        print(f'{target}: no usable labelled trials'); return
    groups, y, X = zip(*selected); y, X, groups = np.asarray(y), np.asarray(X), np.asarray(groups)
    if len(set(groups)) < 3 or len(set(y)) < 2:
        print(f'{target}: need at least three participant files and both label classes'); return
    split = None
    for a, b in GroupShuffleSplit(n_splits=100, test_size=.25, random_state=42).split(X, y, groups):
        if len(np.unique(y[a])) == len(np.unique(y[b])) == 2:
            split = a, b; break
    if split is None:
        print(f'{target}: cannot form participant-held-out train/test sets containing both classes'); return
    a, b = split
    scaler = StandardScaler().fit(X[a])
    clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42).fit(scaler.transform(X[a]), y[a])
    predicted = clf.predict(scaler.transform(X[b]))
    weights = clf.coef_[0] / scaler.scale_
    bias = float(clf.intercept_[0] - np.dot(scaler.mean_, weights))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{target}_weights.npz', weights=weights, bias=bias)
    report = {'target': target, 'train_participants': sorted(set(groups[a])), 'test_participants': sorted(set(groups[b])),
              'train_trials': len(a), 'test_trials': len(b), 'accuracy': accuracy_score(y[b], predicted),
              'balanced_accuracy': balanced_accuracy_score(y[b], predicted),
              'confusion_matrix_rows_actual_columns_predicted_0_1': confusion_matrix(y[b], predicted, labels=[0, 1]).tolist()}
    (out / f'{target}_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{target}: train {len(a)}, test {len(b)}, accuracy {report['accuracy']:.1%}; saved to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recordings', nargs='+', help='One recorded CSV per participant')
    p.add_argument('--out', type=Path, default=Path('models'))
    a = p.parse_args(); rows = get_rows(a.recordings)
    print(f'{len(rows)} usable trials across {len(a.recordings)} recordings')
    if not rows:
        print('The saved weights require 15 seconds of usable EEG per item; the original task shows each item for 4 seconds.')
    for target in ('impulse', 'excitement'):
        train(rows, target, a.out)


if __name__ == '__main__':
    main()
