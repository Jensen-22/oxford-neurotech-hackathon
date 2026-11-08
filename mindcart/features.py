"""Epoch a MindCart session and extract one EEG feature vector per product trial.

    python -m mindcart.features sessions/P01 [--like extreme|median] [--label want|impulse]
    python -m mindcart.features recordings/P01.csv      # one stream_unicorn.py file also works

Per trial:
  baseline = the participant's MEAN fixation power (-0.9..-0.1 s) over all clean trials
  analysis window = 0.5 .. 4.0 s after onset (skips visual onset response / first saccades)
  band power (Welch) in theta 4-8, alpha 8-13, beta 13-30 Hz, as log change vs that person-level baseline
Trials whose analysis window exceeds the artefact threshold (blinks/movement) are flagged.

ERP features (see mindcart/erp.py): epochs -0.2..1.0 s, 0.1-30 Hz, baseline -200..0 ms,
  f_p300_pz/cz = mean 250-500 ms, f_lpp_pz/cpz = mean 500-1000 ms (NaN if the epoch is rejected).

Writes sessions/P01/trials.csv: trial, item_id, rating, label, ok, <reduced feats>, <full feats>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_pipeline import PipelineConfig, filter_offline  # noqa: E402
from mindcart.erp import epochs as erp_epochs, erp_features, erp_filter, out_dir, parse_trials  # noqa: E402
from mindcart.erp import load as load_any  # noqa: E402

FS = 250.0
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BASE_WIN = (-0.9, -0.1)
STIM_WIN = (0.5, 4.0)
POSTERIOR = ["Pz", "PO7", "Oz", "PO8"]
CENTRAL = ["C3", "Cz", "C4"]


def load_session(path):
    """Session folder (eeg.csv + markers.csv) or one stream_unicorn.py CSV."""
    t, x, mk = load_any(path)
    fs = 1.0 / np.median(np.diff(t))
    if abs(fs - FS) > 5:
        print(f"WARNING: estimated sample rate {fs:.1f} Hz (expected {FS})")
    return t, x, parse_trials(mk)


def band_logpower(seg, fs):
    f, p = welch(seg.T, fs=fs, nperseg=min(len(seg), int(fs)), noverlap=int(fs) // 2)
    return {b: np.log(p[:, (f >= lo) & (f < hi)].mean(axis=1) + 1e-12) for b, (lo, hi) in BANDS.items()}


def extract(sdir, like="extreme", ptp_uv=200.0, label="want"):
    cfg = PipelineConfig()
    t, x, trials = load_session(sdir)
    y = filter_offline(x, cfg)  # zero-phase OK offline
    ch = cfg.channel_names
    idx = {c: i for i, c in enumerate(ch)}
    # Pass 1: log band power in each trial's fixation (baseline) and image (stimulus) windows
    raw = []
    for _, tr in trials.iterrows():
        if "onset" not in tr or np.isnan(tr.get("onset", np.nan)):
            continue
        i0 = np.searchsorted(t, tr["onset"])
        sl = lambda w: slice(i0 + int(w[0] * FS), i0 + int(w[1] * FS))
        base, stim = y[sl(BASE_WIN)], y[sl(STIM_WIN)]
        if len(base) < 0.7 * FS or len(stim) < 3 * FS:
            continue
        ptp = float(np.ptp(stim, axis=0).max())
        raw.append((tr, band_logpower(base, FS), band_logpower(stim, FS), ptp))
    # Baseline = this PERSON's average fixation power over all clean trials (a single 0.8 s fixation is too
    # short/noisy for theta and can carry movement from the previous key press). Per-trial ERP baseline is separate.
    clean = [r for r in raw if r[3] < ptp_uv] or raw
    person_base = {b: np.mean([r[1][b] for r in clean], axis=0) for b in BANDS}
    out = []
    for tr, bp_b, bp_s, ptp in raw:
        d = {b: bp_s[b] - person_base[b] for b in BANDS}   # log change vs the person's mean fixation
        rec = {"trial": tr["trial"], "item_id": tr["item_id"], "rating": tr.get("rating", np.nan),
               "need": tr.get("need", np.nan), "max_ptp_uv": ptp, "ok": ptp < ptp_uv}
        # reduced, hypothesis-driven set (8 features; fixed before seeing data)
        mean = lambda b, chans: float(np.mean([d[b][idx[c]] for c in chans]))
        rec.update({
            "f_fz_theta": float(d["theta"][idx["Fz"]]),
            "f_fz_beta": float(d["beta"][idx["Fz"]]),
            "f_post_alpha": mean("alpha", POSTERIOR),
            "f_central_alpha": mean("alpha", CENTRAL),
            "f_asym_c_alpha": float(bp_s["alpha"][idx["C4"]] - bp_s["alpha"][idx["C3"]]),
            "f_asym_po_alpha": float(bp_s["alpha"][idx["PO8"]] - bp_s["alpha"][idx["PO7"]]),
            "f_beta_alpha": float(np.mean(d["beta"] - d["alpha"])),
            "f_post_theta": mean("theta", POSTERIOR),
        })
        for b in BANDS:  # full 24-feature set
            for c in ch:
                rec[f"full_{c}_{b}"] = float(d[b][idx[c]])
        out.append(rec)
    df = pd.DataFrame(out)
    # ERP (P300 / LPP): different filter (0.1-30 Hz) and time-locked epochs
    ep, times, ok_erp = erp_epochs(erp_filter(x), t, trials.set_index("trial").loc[df["trial"], "onset"].to_numpy())
    for k, v in erp_features(ep, times, ok_erp).items():
        df[k] = v
    df["ok_erp"] = ok_erp
    df["label"] = make_labels(df, like, label)
    df.to_csv(out_dir(sdir) / "trials.csv", index=False)
    return df


def make_labels(tr, mode="extreme", label="want"):
    """1 / 0 / NaN (excluded).
    want   : WANT 4-5 -> 1, WANT 1-2 -> 0   (or split at the participant's median with mode="median")
    impulse: WANT>=4 & NEED<=2 -> 1 (impulse);  WANT<=2, or WANT>=4 & NEED>=4 -> 0;  rest excluded"""
    if not isinstance(tr, pd.DataFrame):
        tr = pd.DataFrame({"rating": np.asarray(tr, float)})
    r = tr["rating"].to_numpy(float)
    if label == "impulse":
        n = tr["need"].to_numpy(float) if "need" in tr else np.full(len(r), np.nan)
        return np.where((r >= 4) & (n <= 2), 1.0, np.where((r <= 2) | ((r >= 4) & (n >= 4)), 0.0, np.nan))
    if mode == "extreme":
        return np.where(r >= 4, 1.0, np.where(r <= 2, 0.0, np.nan))
    med = np.nanmedian(r)
    return np.where(r > med, 1.0, np.where(r < med, 0.0, np.nan))  # ties at median dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--like", choices=["extreme", "median"], default="extreme")
    ap.add_argument("--ptp", type=float, default=200.0, help="artefact threshold, µV peak-to-peak")
    ap.add_argument("--label", choices=["want", "impulse"], default="want")
    a = ap.parse_args()
    for s in a.session_dirs:
        df = extract(s, a.like, a.ptp, a.label)
        lab = df["label"]
        print(f"{s}: {len(df)} trials | rejected: band-power {(~df['ok']).mean():.0%}, ERP {(~df['ok_erp']).mean():.0%} | "
              f"like {int((lab == 1).sum())} / dislike {int((lab == 0).sum())} / excluded {int(lab.isna().sum())}")


if __name__ == "__main__":
    main()
