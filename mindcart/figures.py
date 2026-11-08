"""Paper-style figures (EEG-SVRec layout) for UnSway sessions.

    python -m mindcart.figures recordings/P01.csv recordings/P02.csv --items stimuli/items.csv

Writes into <first session>/../figures/:
  fig_distributions.png  rating histograms (WANT, NEED)              ~ EEG-SVRec Fig. 4
  fig_heatmap.png        correlations between ratings, RTs, image     ~ Fig. 5
  fig_topomaps.png       per-channel DE x rating correlation maps     ~ Fig. 6
                         (theta/alpha/beta/gamma; white dot = p < .05)

DE (differential entropy) of a band = 0.5*log(2*pi*e*var) of the band-passed signal in the
0.5-4.0 s image window, minus the participant's mean DE in the fixation (-0.9..-0.1 s).
Correlations are Spearman, computed within each participant on clean trials, then averaged
(Fisher z). With several participants, significance = Stouffer-combined p; with one, its own p.
EXPLORATORY: not part of the pre-registered analysis.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from scipy.stats import norm, spearmanr

from mindcart.erp import CH, FS, load, out_dir

BANDS = {"theta\n4–8 Hz": (4, 8), "alpha\n8–13 Hz": (8, 13), "beta\n13–30 Hz": (13, 30), "gamma\n30–45 Hz": (30, 45)}
WIN, FIX = (0.5, 4.0), (-0.9, -0.1)
PTP_MAX = 200.0
TC_T = np.round(np.arange(-0.9, 3.8001, 0.05), 3)   # avoid the trial edges
BAND_COL = ['#7c3aed', '#2563eb', '#16a34a', '#ea580c']


def trials_with_rt(mk):
    rows = {}
    num = lambda v: float(v) if v not in ("", "None", "nan") else np.nan
    for lt, m in zip(mk["lsl_time"], mk["marker"]):
        p = str(m).split("|")
        if p[0] == "img":
            rows.setdefault(int(p[1]), {}).update(onset=lt, item_id=p[2])
        elif p[0] == "rate" and len(p) >= 4:
            rows.setdefault(int(p[1]), {}).update(WANT=num(p[3]), want_rt=num(p[4]) if len(p) > 4 else np.nan)
        elif p[0] == "need" and len(p) >= 4:
            rows.setdefault(int(p[1]), {}).update(NEED=num(p[3]), need_rt=num(p[4]) if len(p) > 4 else np.nan)
    return pd.DataFrame([{"trial": k, **v} for k, v in sorted(rows.items()) if "onset" in v])


def de_features(path):
    t, x, mk = load(path)
    tr = trials_with_rt(mk)
    sos_bp = butter(4, [1, 45], btype="band", fs=FS, output="sos")
    b, a = iirnotch(50, 30, fs=FS)
    x = sosfiltfilt(np.vstack([sos_bp, tf2sos(b, a)]), x - np.median(x, 0), axis=0)
    de, ok = {}, np.ones(len(tr), bool)
    seg = lambda y, on, w: y[int(np.searchsorted(t, on + w[0])):int(np.searchsorted(t, on + w[1]))]
    for i, on in enumerate(tr.onset):
        s = seg(x, on, WIN)
        ok[i] = len(s) > 0.9 * (WIN[1] - WIN[0]) * FS and np.ptp(s, 0).max() < PTP_MAX
    for bname, (lo, hi) in BANDS.items():
        y = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), x, axis=0)
        img = np.array([0.5 * np.log(2 * np.pi * np.e * seg(y, on, WIN).var(0) + 1e-12) for on in tr.onset])
        fix = np.array([0.5 * np.log(2 * np.pi * np.e * seg(y, on, FIX).var(0) + 1e-12) for on in tr.onset])
        de[bname] = img - np.nanmean(fix[ok], 0)          # participant-level baseline
        # time course: log power in 0.5 s windows every 0.1 s, -1..4 s, vs this trial's fixation
        pw = sosfiltfilt(butter(2, 2, fs=FS, output="sos"), y ** 2, axis=0)   # smoothed instantaneous power
        idx = (TC_T * FS).round().astype(int)
        tc = []
        for on in tr.onset:
            i0 = int(np.searchsorted(t, on))
            if i0 + idx[0] < 0 or i0 + idx[-1] >= len(t):
                tc.append(np.full((len(idx), len(CH)), np.nan)); continue
            lp = np.log(pw[i0 + idx] + 1e-12)
            base = lp[(TC_T >= FIX[0]) & (TC_T <= FIX[1])].mean(0)
            tc.append(lp - base)
        de["_tc_" + bname] = np.array(tc)                   # (trials, times, channels)
    return tr, de, ok


def corr_maps(sessions, target):
    """-> r[band] (8,), p[band] (8,) averaged over participants."""
    zs, ps = {b: [] for b in BANDS}, {b: [] for b in BANDS}
    for tr, de, ok in sessions:
        y = tr[target].to_numpy(float); m = ok & np.isfinite(y)
        if m.sum() < 8 or np.nanstd(y[m]) == 0:
            continue
        for b in BANDS:
            r, p = zip(*[spearmanr(de[b][m, c], y[m]) for c in range(len(CH))])
            zs[b].append(np.arctanh(np.clip(r, -.999, .999)))
            ps[b].append(np.sign(r) * norm.isf(np.array(p) / 2))   # signed z for Stouffer
    out = {}
    for b in BANDS:
        if not zs[b]:
            out[b] = (np.full(len(CH), np.nan), np.ones(len(CH)))
            continue
        r = np.tanh(np.mean(zs[b], 0))
        zc = np.sum(ps[b], 0) / np.sqrt(len(ps[b]))
        out[b] = (r, 2 * norm.sf(np.abs(zc)))
    return out


def fig_topomaps(sessions, targets, path, n_part, note=""):
    import mne
    info = mne.create_info(CH, FS, "eeg"); info.set_montage("standard_1020", on_missing="ignore")
    fig, axes = plt.subplots(len(targets), len(BANDS), figsize=(2.3 * len(BANDS) + 1, 2.3 * len(targets) + .8))
    axes = np.atleast_2d(axes)
    maps = {tg: corr_maps(sessions, tg) for tg in targets}
    vmax = np.nanmax([np.nanmax(np.abs(r)) for m in maps.values() for r, _ in m.values()] + [0.1])
    for i, tg in enumerate(targets):
        for j, b in enumerate(BANDS):
            r, p = maps[tg][b]; ax = axes[i, j]
            if np.all(np.isnan(r)):
                ax.axis("off"); continue
            mask = p < 0.05
            im, _ = mne.viz.plot_topomap(np.nan_to_num(r), info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                                         contours=6, sensors=True, mask=mask,
                                         mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor="k",
                                                          markersize=9))
            if i == 0:
                ax.set_title(b, fontsize=11)
        axes[i, 0].text(-0.35, 0.5, tg, transform=axes[i, 0].transAxes, fontsize=13, ha="right", va="center",
                        fontweight="bold")
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    fig.colorbar(im, cax=cax, label="Spearman r")
    fig.suptitle(f"Correlation of band DE (image 0.5–4 s, vs fixation) with ratings — mean over {n_part} participant(s)\n"
                 "white dot = p < .05 · 8 Unicorn channels, values between electrodes are interpolated" + note,
                 fontsize=10, y=1.0)
    fig.subplots_adjust(left=0.12, right=0.9, wspace=0.05, hspace=0.15)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_band_lines(sessions, path, note=""):
    """4 lines (theta/alpha/beta/gamma): power change vs fixation over time, WANT 4-5 vs WANT 1-2."""
    groups = {"Want it (WANT 4–5)": lambda w: w >= 4, "Don't want it (WANT 1–2)": lambda w: w <= 2}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (gname, sel) in zip(axes, groups.items()):
        for (b, col) in zip(BANDS, BAND_COL):
            per_part = []                                    # mean per participant, then across participants
            for tr, de, ok in sessions:
                m = ok & sel(tr.WANT.to_numpy(float))
                if m.sum() >= 3:
                    per_part.append(np.nanmean(de["_tc_" + b][m].mean(2), 0))   # mean over 8 channels
                    n_tr = m.sum()
            if not per_part:
                continue
            P = np.array(per_part); mu = np.nanmean(P, 0)
            if len(P) > 1:
                se = np.nanstd(P, 0) / np.sqrt(len(P))
            else:
                tr, de, ok = sessions[0]; m = ok & sel(tr.WANT.to_numpy(float))
                d = de["_tc_" + b][m].mean(2); se = np.nanstd(d, 0) / np.sqrt(len(d))
            ax.plot(TC_T, mu, color=col, lw=2, label=b.replace("\n", " "))
            ax.fill_between(TC_T, mu - se, mu + se, color=col, alpha=.15)
        ax.axvspan(*FIX, color="#98a2b3", alpha=.12); ax.axvline(0, color="#475467", lw=1)
        ax.axhline(0, color="#98a2b3", lw=.8); ax.axvspan(*WIN, color="#fde68a", alpha=.15)
        ax.set_title(gname, fontsize=11); ax.set_xlabel("time from outfit onset (s)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("log power change vs fixation")
    axes[0].text(-0.5, axes[0].get_ylim()[1], "fixation", ha="center", va="top", fontsize=8, color="#475467")
    axes[0].text(2.25, axes[0].get_ylim()[1], "analysis window", ha="center", va="top", fontsize=8, color="#92400e")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Band power over time, mean of 8 channels ± SEM, {len(sessions)} participant(s)" + note, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_distributions(df, path, note=""):
    cols = [("WANT", "#8fd0c2"), ("NEED", "#f6c945")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, (c, col) in zip(axes, cols):
        v = df[c].dropna().astype(int)
        ax.bar(range(1, 6), [int((v == k).sum()) for k in range(1, 6)], width=.45, color=col)
        ax.set_title(c); ax.set_xticks(range(1, 6))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("trials")
    fig.suptitle("Distribution of ratings (all participants)" + note, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_heatmap(df, path, note=""):
    cols = [c for c in ["WANT", "NEED", "want_rt", "need_rt", "luminance", "contrast", "colourfulness"]
            if c in df and df[c].notna().sum() > 5 and df[c].std() > 0]
    keep = []                       # drop a column that duplicates an earlier one (e.g. identical RTs)
    for c in cols:
        if not any(df[c].corr(df[k]) > 0.999 for k in keep):
            keep.append(c)
    cols = keep
    r = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1 + .75 * len(cols), .6 + .65 * len(cols)))
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right"); ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{r.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(r.iloc[i, j]) > .6 else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("Spearman correlations: ratings, reaction times, image properties" + note, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--items", default="stimuli/items.csv")
    ap.add_argument("--note", default="", help="extra title text, e.g. 'SIMULATED DATA'")
    a = ap.parse_args()
    note = f"\n{a.note}" if a.note else ""
    items = pd.read_csv(a.items); items["item_id"] = items["item_id"].astype(str)
    sessions, rows = [], []
    for s in a.sessions:
        tr, de, ok = de_features(s)
        tr["impulse"] = ((tr.WANT >= 4) & (tr.NEED <= 2)).astype(float)
        sessions.append((tr, de, ok))
        rows.append(tr[ok].assign(participant=Path(s).stem).merge(items, on="item_id", how="left"))
        print(f"{s}: {len(tr)} trials, {ok.sum()} clean ({100 * (1 - ok.mean()):.0f}% rejected)")
    df = pd.concat(rows)
    out = out_dir(a.sessions[0]).parent / "figures"; out.mkdir(exist_ok=True)
    fig_distributions(df, out / "fig_distributions.png", note)
    fig_heatmap(df, out / "fig_heatmap.png", note)
    fig_band_lines(sessions, out / "fig_band_lines.png", note)
    fig_topomaps(sessions, ["WANT", "NEED", "impulse"], out / "fig_topomaps.png", len(sessions), note)
    print(f"Saved figures in {out}")


if __name__ == "__main__":
    main()
