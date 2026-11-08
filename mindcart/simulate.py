"""Simulate full MindCart sessions (items, prefs, EEG + markers) with a known ground truth.

    python -m mindcart.simulate --out sim --participants 2 --n-items 70 --effect 0.5
    python -m mindcart.simulate --out sim_null --effect 0      # no EEG effect -> B should be at chance
    python -m mindcart.simulate --out sim --oddball            # oddball P300 validation session

Liked items: Fz theta up, posterior alpha down during viewing (scaled by --effect).
Bright images also lower occipital alpha (the confound Model A must absorb).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS = 250
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CATS = ["tshirt", "jacket", "dress", "jeans", "hoodie", "shirt"]
COLS = ["black", "white", "blue", "red", "green", "beige"]


def pink(n, rng, n_ch):
    w = rng.standard_normal((n_ch, n)); s = np.fft.rfft(w, axis=1); f = np.fft.rfftfreq(n, 1 / FS)
    s /= np.sqrt(np.maximum(f, 1.0)); x = np.fft.irfft(s, n=n, axis=1)
    return (x / x.std(axis=1, keepdims=True)).T


# ERP topography weights (P300/LPP are centro-parietal)
ERP_W = {"Fz": .3, "C3": .5, "Cz": .8, "C4": .5, "Pz": 1.0, "PO7": .6, "Oz": .4, "PO8": .6}


def erp_wave(n_samp, p300_uv, lpp_uv, latency=0.35):
    """Single-trial ERP: P300 gaussian peak + sustained LPP (500-1000 ms), plus small jitter in latency."""
    tt = np.arange(n_samp) / FS
    p3 = p300_uv * np.exp(-0.5 * ((tt - latency) / 0.07) ** 2)
    lpp = lpp_uv * (1 / (1 + np.exp(-(tt - 0.45) / 0.04))) * (1 / (1 + np.exp((tt - 1.1) / 0.08)))
    return p3 + lpp


def make_items(n, rng):
    return pd.DataFrame({
        "item_id": [f"I{i:03d}" for i in range(n)], "file": [f"I{i:03d}.jpg" for i in range(n)],
        "category": rng.choice(CATS, n), "colour": rng.choice(COLS, n),
        "price": np.round(rng.uniform(10, 120, n), 2), "luminance": rng.uniform(0.3, 0.7, n),
        "contrast": rng.uniform(0.1, 0.3, n), "colourfulness": rng.uniform(0.05, 0.5, n)})


def simulate_participant(items, sdir, rng, effect):
    sdir.mkdir(parents=True, exist_ok=True)
    pc = {c: int(rng.integers(1, 6)) for c in CATS}; pcol = {c: int(rng.integers(1, 6)) for c in COLS}
    pd.DataFrame([("category", k, v) for k, v in pc.items()] + [("colour", k, v) for k, v in pcol.items()],
                 columns=["type", "value", "score"]).to_csv(sdir / "prefs.csv", index=False)
    order = rng.permutation(len(items))
    # latent liking: partly predictable from prefs/price, partly idiosyncratic (only EEG can see it)
    pref = np.array([pc[items.category[i]] + pcol[items.colour[i]] for i in order], float)
    idio = rng.standard_normal(len(order))
    latent = 0.35 * (pref - pref.mean()) / pref.std() - 0.2 * (items.price.values[order] - 65) / 30 + 0.9 * idio
    rating = np.clip(np.round(3 + 1.3 * latent + rng.normal(0, 0.4, len(order))), 1, 5).astype(int)
    need = np.clip(np.round(2.6 + 0.4 * latent + rng.normal(0, 1.1, len(order))), 1, 5).astype(int)

    trial_s = 1.0 + 4.0 + 2.5 + 0.7
    n = int((len(order) * trial_s + 10) * FS)
    x = 8.0 * pink(n, rng, len(CH))
    t = np.arange(n) / FS + 1000.0  # fake LSL clock
    alpha_base = 6.0
    post = [CH.index(c) for c in ("Pz", "PO7", "Oz", "PO8")]
    markers, cur = [("start", t[int(2 * FS)])], 3.0
    phase = lambda: rng.uniform(0, 2 * np.pi)
    for k, i in enumerate(order):
        fix0 = int(cur * FS); on = fix0 + FS; off = on + 4 * FS
        markers.append((f"fix|{k}", t[fix0])); markers.append((f"img|{k}|{items.item_id[i]}", t[on]))
        tt = np.arange(off - fix0) / FS
        for c in post:  # alpha present in fixation; during viewing reduced by liking + brightness
            amp = np.full(len(tt), alpha_base)
            amp[FS:] = alpha_base * (1 - 0.25 - effect * 0.25 * np.tanh(idio[k]) - 0.6 * (items.luminance[i] - 0.5))
            x[fix0:off, c] += amp * np.sin(2 * np.pi * 10 * tt + phase())
        th = np.zeros(len(tt)); th[FS:] = effect * 3.0 * max(np.tanh(idio[k]), 0) + 1.0
        x[fix0:off, CH.index("Fz")] += th * np.sin(2 * np.pi * 6 * tt + phase())
        # ERP to image onset: everyone gets a P300; liked items get larger P300 and a sustained LPP
        lk = np.tanh(idio[k])
        w = erp_wave(int(1.5 * FS), p300_uv=4 + effect * 1.5 * max(lk, 0), lpp_uv=1 + effect * 3.0 * lk,
                     latency=0.35 + rng.normal(0, 0.02))
        for c, wt in ERP_W.items():
            x[on:on + len(w), CH.index(c)] += wt * w
        if rng.random() < 0.08:  # occasional blink in the viewing window
            b = on + int(rng.uniform(0.6, 3.5) * FS); x[b:b + 60, 0] += 300 * np.hanning(60)
        rt = round(float(rng.uniform(0.6, 1.6)), 3)
        markers.append((f"rate|{k}|{items.item_id[i]}|{rating[k]}|{rt}", t[off] + rt))
        markers.append((f"need|{k}|{items.item_id[i]}|{need[k]}|{rt}", t[off] + rt + 1.0))
        cur += trial_s + rng.uniform(-0.2, 0.2)
    markers.append(("end", t[int(cur * FS)]))
    x += rng.normal(0, 3000, len(CH))  # DC offsets
    pd.DataFrame(np.column_stack([t, x]), columns=["lsl_time"] + [f"EEG {j + 1}" for j in range(len(CH))]) \
        .to_csv(sdir / "eeg.csv", index=False, float_format="%.4f")
    pd.DataFrame(markers, columns=["marker", "lsl_time"])[["lsl_time", "marker"]].to_csv(sdir / "markers.csv", index=False)
    return rating


def to_single_file(sdir, path):
    """Convert a simulated session folder into stream_unicorn.py's one-file format (markers on samples)."""
    eeg = pd.read_csv(sdir / "eeg.csv"); mk = pd.read_csv(sdir / "markers.csv")
    col = [""] * len(eeg); t = eeg["lsl_time"].to_numpy()
    for lt, m in zip(mk["lsl_time"], mk["marker"]):
        i = min(int(np.searchsorted(t, lt)), len(t) - 1)
        col[i] = (col[i] + ";" if col[i] else "") + m
    eeg["marker"] = col
    eeg.to_csv(path, index=False, float_format="%.4f")


def simulate_oddball(sdir, rng, n=200, p_target=0.2, p300_uv=6.0):
    """Oddball session: 200 ms stimuli every ~1 s, 20% targets elicit a P300."""
    sdir.mkdir(parents=True, exist_ok=True)
    n_samp = int((n * 1.1 + 6) * FS)
    x = 8.0 * pink(n_samp, rng, len(CH)); t = np.arange(n_samp) / FS + 2000.0
    kinds = np.where(rng.random(n) < p_target, "target", "standard")
    markers, cur = [("start", t[int(1 * FS)])], 2.0
    for k, kind in enumerate(kinds):
        on = int(cur * FS)
        markers.append((f"odd|{k}|{kind}", t[on]))
        w = erp_wave(int(0.8 * FS), p300_uv if kind == "target" else 1.0, 0, latency=0.33 + rng.normal(0, 0.02))
        for c, wt in ERP_W.items():
            x[on:on + len(w), CH.index(c)] += wt * w
        cur += rng.uniform(0.9, 1.2)
    markers.append(("end", t[int(cur * FS)]))
    x += rng.normal(0, 3000, len(CH))
    pd.DataFrame(np.column_stack([t, x]), columns=["lsl_time"] + [f"EEG {j + 1}" for j in range(len(CH))]) \
        .to_csv(sdir / "eeg.csv", index=False, float_format="%.4f")
    pd.DataFrame(markers, columns=["marker", "lsl_time"])[["lsl_time", "marker"]].to_csv(sdir / "markers.csv", index=False)
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim")
    ap.add_argument("--participants", type=int, default=2)
    ap.add_argument("--n-items", type=int, default=70)
    ap.add_argument("--effect", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oddball", action="store_true", help="simulate an oddball P300 validation session instead")
    ap.add_argument("--single-file", action="store_true", help="also write each session as one stream_unicorn.py CSV")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.oddball:
        k = simulate_oddball(out / "OB01", rng)
        print(f"Wrote {out}/OB01: {int((k == 'target').sum())} targets, {int((k == 'standard').sum())} standards")
        return
    items = make_items(a.n_items, rng); items.to_csv(out / "items.csv", index=False)
    for p in range(a.participants):
        r = simulate_participant(items, out / f"P{p + 1:02d}", rng, a.effect)
        if a.single_file:
            to_single_file(out / f"P{p + 1:02d}", out / f"P{p + 1:02d}_stream.csv")
        print(f"P{p + 1:02d}: ratings {np.bincount(r, minlength=6)[1:]} (counts of 1..5)")
    print(f"Wrote {out}/items.csv and {a.participants} sessions")


if __name__ == "__main__":
    main()
