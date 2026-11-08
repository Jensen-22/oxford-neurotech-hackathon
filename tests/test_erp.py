import numpy as np

import mindcart.simulate as sim
from mindcart.erp import CH, epochs, erp_features, erp_filter, oddball_erp


def test_epoch_baseline_and_window_means():
    fs = 250
    t = np.arange(10 * fs) / fs
    x = np.zeros((len(t), 8))
    on = 5.0
    i0 = int(on * fs)
    x[i0 + int(.30 * fs):i0 + int(.40 * fs), CH.index("Pz")] = 10.0   # P300-like bump
    x[i0 + int(.60 * fs):i0 + int(.90 * fs), CH.index("Cz")] = 4.0    # LPP-like shift
    ep, times, ok = epochs(x, t, [on])
    assert ok[0] and ep.shape == (1, 300, 8)
    f = erp_features(ep, times, ok)
    assert f["f_p300_pz"][0] > 3 and abs(f["f_p300_cz"][0]) < 1e-9
    assert f["f_lpp_czpz"][0] > 0.5


def test_artefact_epoch_rejected():
    fs = 250
    t = np.arange(4 * fs) / fs
    x = np.zeros((len(t), 8)); x[int(2.3 * fs), 0] = 500  # blink-sized spike on Fz
    ep, times, ok = epochs(x, t, [2.0])
    assert not ok[0] and np.isnan(erp_features(ep, times, ok)["f_p300_pz"][0])


def test_erp_filter_keeps_slow_wave():
    fs = 250
    t = np.arange(20 * fs) / fs
    w = np.zeros(len(t)); w[10 * fs:10 * fs + fs // 2] = 5.0         # 500 ms positive shift
    y = erp_filter(np.tile(w[:, None], (1, 8)))
    assert y[10 * fs + fs // 4, 0] > 3.0                          # a 1 Hz high-pass would kill this


def test_oddball_check_detects_p300(tmp_path):
    rng = np.random.default_rng(4)
    sim.simulate_oddball(tmp_path / "OB", rng)
    diff, p, verdict, nt, ns = oddball_erp(tmp_path / "OB")
    assert diff > 1.5 and p < 0.05 and verdict.startswith("PASS")
