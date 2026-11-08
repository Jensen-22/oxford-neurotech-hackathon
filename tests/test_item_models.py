import tempfile
import unittest
from pathlib import Path

import numpy as np

from mindcart.live_models import ItemModels, window_features
from mindcart.train_layers import label, train


class ItemModelsTest(unittest.TestCase):
    def test_item_timing_and_prediction(self):
        rng = np.random.default_rng(12)
        t = np.arange(0, 23, .004)
        x = rng.normal(0, 8, (len(t), 8))
        model = {'weights': np.zeros(21), 'bias': 1.0}
        adapter = ItemModels({'video_liking': model, 'impulse': model})
        # Deliver each marker on time; processing holds one second for LSL transport.
        for start in range(0, len(t), 100):
            end = min(start + 100, len(t))
            markers = []
            if t[start] <= 1 < t[end-1]: markers.append((1.0, 'img|1|jacket'))
            if t[start] <= 20 < t[end-1]: markers.append((20.0, 'view_end|1|jacket'))
            adapter.push(x[start:end], t[start:end], markers)
        self.assertEqual(adapter.state['item_id'], 'jacket')
        # The original four-second item cannot supply three five-second windows.
        self.assertEqual(adapter.state['status'], 'insufficient_eeg')
        self.assertEqual(adapter.state['usable_windows'], 0)
        self.assertIsNone(adapter.state['video_liking'])
        self.assertIsNone(adapter.state['impulse'])
        self.assertIsNone(adapter.state['excitement'])
        adapter.event('rate|1|jacket|5|1.2')
        adapter.event('need|1|jacket|1|1.4')
        self.assertEqual(adapter.state['rated_impulse'], 1)
        self.assertIsNone(adapter.state['impulse'])

    def test_short_or_interrupted_items_are_not_predicted(self):
        rng = np.random.default_rng(9)
        t = np.arange(0, 8, .004); x = rng.normal(0, 8, (len(t), 8))
        adapter = ItemModels({'video_liking': {'weights': np.zeros(21), 'bias': 1.0}})
        adapter.push(x, t, [(1, 'img|1|shirt'), (5, 'view_end|1|shirt')])
        self.assertEqual(adapter.state['status'], 'insufficient_eeg')
        self.assertIsNone(adapter.state['video_liking'])
        flat = x[:1250].copy(); flat[:, 2] = 0
        self.assertIsNone(window_features(flat))

    def test_labels_and_participant_split(self):
        self.assertEqual(label({'rating': 5, 'need': 1}, 'impulse'), 1)
        self.assertEqual(label({'rating': 5, 'need': 5}, 'impulse'), 0)
        self.assertIsNone(label({'rating': 3, 'need': 3}, 'impulse'))
        self.assertEqual(label({'excitement': 1}, 'excitement'), 0)
        rows = []
        for person in range(4):
            for outcome in (0, 1):
                for k in range(3):
                    tr = {'rating': 5 if outcome else 1, 'need': 1, 'excitement': 5 if outcome else 1}
                    rows.append((f'person{person}', tr, np.full(21, outcome + person*.01 + k*.001)))
        with tempfile.TemporaryDirectory() as tmp:
            train(rows, 'impulse', Path(tmp))
            import json
            report = json.loads((Path(tmp) / 'impulse_evaluation.json').read_text())
            self.assertFalse(set(report['train_participants']) & set(report['test_participants']))
            self.assertTrue((Path(tmp) / 'impulse_weights.npz').exists())


if __name__ == '__main__':
    unittest.main()
