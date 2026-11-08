"""Event-related potentials: P300 and LPP.

Both are positive waves over centro-parietal scalp after a stimulus:
  P300 (P3b)  250-500 ms, max at Pz   - attention / "this is relevant"
  LPP         500-1000 ms, Pz + Cz    - sustained motivational / emotional significance (interest, liking)

ERP processing differs from the band-power pipeline: slow waves need a 0.1 Hz high-pass (not 1 Hz),
epochs are time-locked to the image-onset marker, baseline-corrected, artefact-rejected and averaged.

    python -m mindcart.erp sessions/P01                 # product task: want vs don't-want ERP plot
    python -m mindcart.erp recordings/P01.csv --label impulse   # single stream_unicorn.py file works too
    python -m mindcart.erp sessions/OB01 --oddball      # oddball validation: target vs standard P300
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

FS = 250.0
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
EPOCH = (-0.2, 1.0)          # s around onset
BASELINE = (-0.2, 0.0)
P300_WIN = (0.25, 0.50)
LPP_WIN = (0.50, 1.00)
REJECT_UV = 100.0            # |amplitude| limit in any ERP channel after baseline
ERP_CH = ["Fz", "Cz", "Pz"]  # channels used for rejection (blinks show at Fz)


def erp_filter(x, fs=FS):
    """0.1-30 Hz band-pass + 50 Hz notch, zero-phase (offline). Keeps slow ERP components intact."""
    sos = butter(2, [0.1, 30], btype="band", fs=fs, output="sos")
    b, a = iirnotch(50, 30, fs=fs)
    sos = np.vstack([sos, tf2sos(b, a)])
    return sosfiltfilt(sos, x - np.median(x, axis=0), axis=0)


def epochs(y, t, onsets, fs=FS, win=EPOCH, base=BASELINE):
    """y (n, ch) filtered, t (n,) timestamps, onsets list -> (n_trials, n_times, ch), times, ok mask."""
    n0, n1 = int(round(win[0] * fs)), int(round(win[1] * fs))
    times = np.arange(n0, n1) / fs
    b = (times >= base[0]) & (times < base[1])
    out, ok = [], []
    rej_idx = [CH.index(c) for c in ERP_CH]
    for on in onsets:
        i0 = int(np.searchsorted(t, on))
        if i0 + n0 < 0 or i0 + n1 > len(y):
            out.append(np.full((n1 - n0, y.shape[1]), np.nan)); ok.append(False); continue
        e = y[i0 + n0:i0 + n1].copy()
        e -= e[b].mean(axis=0)
        out.append(e)
        ok.append(bool(np.abs(e[:, rej_idx]).max() < REJECT_UV))
    return np.asarray(out), times, np.asarray(ok)


def window_mean(ep, times, win, ch_names):
    m = (times >= win[0]) & (times < win[1])
    idx = [CH.index(c) for c in ch_names]
    return ep[:, m][:, :, idx].mean(axis=(1, 2))


def erp_features(ep, times, ok):
    """Per-trial P300/LPP mean amplitudes (µV). NaN for rejected trials."""
    f = {
        "f_p300_pz": window_mean(ep, times, P300_WIN, ["Pz"]),
        "f_p300_cz": window_mean(ep, times, P300_WIN, ["Cz"]),
        "f_lpp_pz": window_mean(ep, times, LPP_WIN, ["Pz"]),
        "f_lpp_czpz": window_mean(ep, times, LPP_WIN, ["Cz", "Pz"]),
    }
    for k in f:
        f[k] = np.where(ok, f[k], np.nan)
    return f


def out_dir(path):
    """Where results go: the session folder, or <file>/ next to a single-file recording."""
    p = Path(path)
    if p.is_file():
        d = p.with_suffix(""); d.mkdir(exist_ok=True); return d
    return p


def load(path):
    """Accepts EITHER a session folder (eeg.csv + markers.csv from record_session.py)
    OR one CSV from stream_unicorn.py (lsl_time, EEG 1..8, marker). Returns t, x, markers-df."""
    p = Path(path)
    if p.is_file():
        df = pd.read_csv(p, dtype={"marker": str}, low_memory=False)
        eeg_cols = [c for c in df.columns if c.strip().upper().startswith("EEG")]
        t = df["lsl_time"].to_numpy(float)
        rows = [(tt, m) for tt, ms in zip(t, df["marker"]) if isinstance(ms, str) and ms for m in ms.split(";")]
        mk = pd.DataFrame(rows, columns=["lsl_time", "marker"])
        return t, df[eeg_cols].to_numpy(float), mk
    eeg = pd.read_csv(p / "eeg.csv")
    mk = pd.read_csv(p / "markers.csv", dtype={"marker": str})
    return eeg["lsl_time"].to_numpy(), eeg.drop(columns="lsl_time").to_numpy(float), mk


def parse_trials(mk):
    """Markers -> one row per trial: onset, item_id, rating (= WANT), need."""
    rows = {}
    num = lambda v: float(v) if v not in ("", "None", "nan") else np.nan
    for lt, m in zip(mk["lsl_time"], mk["marker"]):
        p = str(m).split("|")
        if p[0] == "img":
            rows.setdefault(int(p[1]), {}).update(onset=lt, item_id=p[2])
        elif p[0] == "rate":
            rows.setdefault(int(p[1]), {})["rating"] = num(p[3])
        elif p[0] == "need":
            rows.setdefault(int(p[1]), {})["need"] = num(p[3])
    return pd.DataFrame([{"trial": k, **v} for k, v in sorted(rows.items()) if "onset" in v])


def plot_erp(times, groups, title, path, note=""):
    """groups: {label: (epochs, color)} -> Pz and Cz averages with ±SEM and shaded P300/LPP windows."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ms = times * 1000
    for ax, ch in zip(axes, ["Pz", "Cz"]):
        ax.axvspan(P300_WIN[0] * 1000, P300_WIN[1] * 1000, color="#3e4fd8", alpha=.07)
        ax.axvspan(LPP_WIN[0] * 1000, LPP_WIN[1] * 1000, color="#12805c", alpha=.07)
        ax.axvline(0, color="#98a2b3", lw=1); ax.axhline(0, color="#98a2b3", lw=.8)
        for lab, (ep, col) in groups.items():
            if len(ep) == 0:
                continue
            d = ep[:, :, CH.index(ch)]
            mu, se = np.nanmean(d, 0), np.nanstd(d, 0) / np.sqrt(max(1, len(d)))
            ax.plot(ms, mu, color=col, lw=2, label=f"{lab} (n={len(d)})")
            ax.fill_between(ms, mu - se, mu + se, color=col, alpha=.18)
        ax.set_title(ch, loc="left", fontweight="bold"); ax.set_xlabel("ms after image onset")
        ax.text(P300_WIN[0] * 1000 + 5, ax.get_ylim()[1] * .92 if ax.get_ylim()[1] > 0 else 1, "P300", color="#3e4fd8", fontsize=9)
        ax.text(LPP_WIN[0] * 1000 + 5, ax.get_ylim()[1] * .92 if ax.get_ylim()[1] > 0 else 1, "LPP", color="#12805c", fontsize=9)
    axes[0].set_ylabel("µV (positive up)"); axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title + (f"\n{note}" if note else ""), x=.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def product_erp(sdir, like="extreme", label="want"):
    from .features import make_labels
    t, x, mk = load(sdir)
    tr = parse_trials(mk)
    y = erp_filter(x)
    ep, times, ok = epochs(y, t, tr["onset"].to_numpy())
    lab = make_labels(tr, like, label)
    f = erp_features(ep, times, ok)
    like_m, dis_m = ok & (lab == 1), ok & (lab == 0)
    d = {k: (np.nanmean(v[like_m]) - np.nanmean(v[dis_m])) for k, v in f.items()}
    names = {"want": ("want it (4-5)", "don't want (1-2)"), "impulse": ("impulse: want↑ need↓", "no impulse")}[label]
    plot_erp(times, {names[0]: (ep[like_m], "#12805c"), names[1]: (ep[dis_m], "#98a2b3")},
             f"{Path(sdir).stem}: ERP by {label}",
             out_dir(sdir) / f"erp_{label}.png",
             f"{ok.sum()}/{len(ok)} trials clean · difference: P300(Pz) {d['f_p300_pz']:+.1f} µV, LPP(Cz/Pz) {d['f_lpp_czpz']:+.1f} µV")
    return d, ok


def oddball_erp(sdir):
    t, x, mk = load(sdir)
    ev = [(lt, m.split("|")[2]) for lt, m in zip(mk["lsl_time"], mk["marker"]) if str(m).startswith("odd|")]
    if len(ev) < 10:
        msg = (f"Only {len(ev)} oddball markers in this recording. The experiment's markers didn't reach the "
               "recorder (was stream_unicorn.py running BEFORE the task started, and did its 'mk' counter go up?)")
        print("ERROR: " + msg)
        return float("nan"), 1.0, "FAIL: no markers", 0, 0
    on = np.array([e[0] for e in ev]); kind = np.array([e[1] for e in ev])
    y = erp_filter(x)
    ep, times, ok = epochs(y, t, on, win=(-0.2, 0.8))
    # PASS rule (fixed in advance, see ANALYSIS_PLAN.md): mean amplitude at Pz, 250-500 ms,
    # target - standard > 0 with a one-sided trial-level permutation test p < .05 (5000 label shuffles).
    p3 = window_mean(ep, times, P300_WIN, ["Pz"])
    tg, st = ok & (kind == "target"), ok & (kind == "standard")
    vals, is_t = p3[ok], (kind[ok] == "target")
    diff = vals[is_t].mean() - vals[~is_t].mean()
    rng = np.random.default_rng(0)
    null = np.array([(lambda m: vals[m].mean() - vals[~m].mean())(rng.permutation(is_t)) for _ in range(5000)])
    p = (np.sum(null >= diff) + 1) / (len(null) + 1)
    tstat = diff / (null.std() + 1e-12)
    verdict = "PASS: clear P300, the setup records ERPs" if (diff > 0 and p < .05) else "WEAK / FAIL: check markers, contact, or run more trials"
    plot_erp(times, {"target (rare)": (ep[tg], "#3e4fd8"), "standard": (ep[st], "#98a2b3")},
             f"{Path(sdir).stem}: oddball P300 check", out_dir(sdir) / "oddball_p300.png",
             f"P300 at Pz (250-500 ms): target − standard = {diff:+.1f} µV (permutation test, p = {p:.2g}) → {verdict}"
             f"  · {(~ok).mean():.0%} epochs rejected")
    print(f"  rejected epochs: {(~ok).sum()}/{len(ok)} ({(~ok).mean():.0%})")
    return diff, p, verdict, int(tg.sum()), int(st.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--oddball", action="store_true")
    ap.add_argument("--like", choices=["extreme", "median"], default="extreme")
    ap.add_argument("--label", choices=["want", "impulse"], default="want")
    a = ap.parse_args()
    for s in a.session_dirs:
        if a.oddball:
            diff, p, v, nt, ns = oddball_erp(s)
            print(f"{s}: targets {nt}, standards {ns} | P300(Pz) diff {diff:+.1f} µV, p={p:.2g} | {v}\n  -> {out_dir(s) / 'oddball_p300.png'}")
        else:
            d, ok = product_erp(s, a.like, a.label)
            print(f"{s}: {ok.sum()}/{len(ok)} clean trials | {a.label} vs not: "
                  + ", ".join(f"{k[2:]} {v:+.2f} µV" for k, v in d.items()) + f"\n  -> {out_dir(s) / f'erp_{a.label}.png'}")


if __name__ == "__main__":
    main()
