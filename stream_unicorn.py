#!/usr/bin/env python3
"""stream_unicorn.py: ONE file to stream + record the g.tec Unicorn.

  * finds the Unicorn: Bluetooth serial (macOS/Linux, no g.tec software), the "Unicorn LSL" app, or UnicornPy (Windows)
  * records EEG + any experiment markers into ONE CSV:  recordings/unicorn_<time>.csv
        columns: lsl_time, EEG 1..8, marker
  * prints live signal quality every second
  * serves band-power features on ws://<this-laptop>:8765 for dashboard/index.html and mindcart.html
  * serves raw EEG on ws://<this-laptop>:8766 for dashboard/pipeline.html (live step-by-step pipeline board)

Install once:   pip install numpy scipy pylsl websockets pyserial
Mac:            pair "UN-xxxx" in System Settings > Bluetooth, then just run the commands below
Test device:    python stream_unicorn.py --check         (~60 s guided test -> PASS/FAIL)
Run:            python stream_unicorn.py                 (auto: Bluetooth serial -> LSL -> UnicornPy)
                python stream_unicorn.py --source unicornpy
                python stream_unicorn.py --fake          (no headset, for testing)
Stop:           Ctrl+C  (file is saved continuously)
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch
from mindcart.live_models import ItemModels

FS = 250.0
N_EEG = 8
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
WIN, STEP = int(2 * FS), int(1 * FS)


# ----------------------------------------------------------------- sources
class MarkerListener:
    """Finds the experiment's LSL marker stream (type 'Markers') whenever it appears, in a background
    thread, and hands its markers (on the local LSL clock) to the recorder. Used for every source."""
    def __init__(self):
        self.inlet, self.found = None, threading.Event()
        threading.Thread(target=self._find, daemon=True).start()

    def _find(self):
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop
        while self.inlet is None:
            m = resolve_byprop("type", "Markers", timeout=2.0)
            if m:
                self.inlet = StreamInlet(m[0], max_buflen=360, processing_flags=proc_clocksync | proc_dejitter)
                self.inlet.open_stream(timeout=5.0)
                print(f"\n[markers] experiment stream connected: {m[0].name()}", flush=True)
                self.found.set()

    def poll(self):
        if self.inlet is None:
            return []
        md, mt = self.inlet.pull_chunk(timeout=0.0)
        return [(t, str(d[0])) for d, t in zip(md, mt)]


class LSLSource:
    def __init__(self, timeout=8):
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_streams
        streams = resolve_streams(wait_time=timeout)
        eeg = [s for s in streams if s.channel_count() >= N_EEG and s.nominal_srate() > 0 and
               ("unicorn" in s.name().lower() or s.type().lower() in ("eeg", "data"))]
        if not eeg:
            names = [f"{s.name()} ({s.type()}, {s.channel_count()} ch)" for s in streams]
            raise RuntimeError("No Unicorn LSL stream. Open the 'Unicorn LSL' app, pick the headset, "
                               f"click Open then Start. Streams seen: {names or 'none'}")
        s = eeg[0]
        print(f"[source] LSL: {s.name()}  type={s.type()}  {s.channel_count()} ch @ {s.nominal_srate():.0f} Hz")
        self.inlet = StreamInlet(s, max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)

    def read(self):
        data, ts = self.inlet.pull_chunk(timeout=0.1, max_samples=256)
        mks = []
        return (np.asarray(data, float)[:, :N_EEG] if data else np.empty((0, N_EEG))), np.asarray(ts), mks


class UnicornPySource:
    """Direct USB-dongle access via g.tec's UnicornPy (Windows, ships with Unicorn Suite)."""
    def __init__(self):
        import UnicornPy
        devs = UnicornPy.GetAvailableDevices(True)
        if not devs:
            raise RuntimeError("No Unicorn paired. Pair it in Unicorn Suite first.")
        print(f"[source] UnicornPy: {devs[0]}")
        self.dev = UnicornPy.Unicorn(devs[0])
        self.n_ch = self.dev.GetNumberOfAcquiredChannels()
        self.n = 25
        self.buf = bytearray(self.n * self.n_ch * 4)
        self.dev.StartAcquisition(False)
        from pylsl import local_clock
        self.t0 = local_clock()
        self.count = 0

    def read(self):
        self.dev.GetData(self.n, self.buf, len(self.buf))
        x = np.frombuffer(self.buf, dtype=np.float32).reshape(self.n, self.n_ch)[:, :N_EEG].astype(float)
        ts = self.t0 + (self.count + np.arange(self.n)) / FS
        self.count += self.n
        return x, ts, []

    def close(self):
        self.dev.StopAcquisition()


class SerialSource:
    """macOS/Linux: read the Unicorn directly over its Bluetooth serial port (no g.tec software needed).

    Pair "UN-xxxx.xx.xx" in System Settings > Bluetooth first. It shows up as /dev/cu.UN-*.
    Frame = 45 bytes: C0 00 | battery | 8x EEG int24 BE | 3x acc int16 LE | 3x gyro int16 LE | counter u32 LE | 0D 0A
    Also republishes everything as an LSL stream "Unicorn" (type EEG) so other tools can use it.
    """
    START, STOP = bytes([0x61, 0x7C, 0x87]), bytes([0x63, 0x5C, 0xC5])
    FRAME = 45
    UV = 4500000.0 / 50331642.0

    @classmethod
    def _probe(cls, serial, port, wait=15.0):
        """Open the port, send START (re-sent every 4 s), return the open Serial if bytes arrive within `wait` s."""
        try:
            s = serial.Serial(port, 115200, timeout=0.5)
            s.reset_input_buffer()
            s.write(cls.STOP)          # clear a stream left running by a crashed session
            time.sleep(0.3)
            s.reset_input_buffer()
            s.write(cls.START)
            t0 = last = time.time()
            while time.time() - t0 < wait:
                if s.read(cls.FRAME):
                    return s
                if time.time() - last > 4:
                    print(f"[source]   still waiting for {port} ({int(time.time() - t0)} s)…", flush=True)
                    s.write(cls.START); last = time.time()
            try:
                s.write(cls.STOP)
            except Exception:
                pass
            s.close()
        except Exception as e:
            print(f"[source]   {port}: {e}")
        return None

    def __init__(self, port=None, lsl_out=True):
        import glob
        import serial  # pip install pyserial
        port = port or os.environ.get("UNSWAY_SERIAL")
        if port is None:
            # newest headset first (UN-2026… before UN-2023…); several may be paired
            ports = sorted(set(glob.glob("/dev/cu.UN-*") + glob.glob("/dev/rfcomm*")), reverse=True)
            if not ports:
                raise RuntimeError("No Unicorn serial port (/dev/cu.UN-*). Pair the headset in Bluetooth settings first.")
            print(f"[source] paired Unicorns: {', '.join(ports)}")
            self.ser = None
            for p in ports:
                print(f"[source] trying {p} …  (the headset must be ON; its LED stops blinking when connected)")
                s = self._probe(serial, p)
                if s is not None:
                    self.ser, port = s, p
                    break
                print(f"[source]   no data from {p} (off, out of range, or a different headset)")
            if self.ser is None:
                raise RuntimeError("None of the paired Unicorns sent data. Turn the headset on (LED blinking), "
                                   "close any other app/laptop using it, and try again.")
        else:
            self.ser = self._probe(serial, port)
            if self.ser is None:
                raise RuntimeError(f"No data from {port}. Is the headset on and not used by another computer?")
        print(f"[source] Bluetooth serial: {port}  CONNECTED, data flowing")
        self.buf = bytearray()
        self.last_counter = None
        self.dropped = 0
        import collections
        self._obs, self.offset, self.n_samples = collections.deque(), None, 0
        self.outlet = None
        if lsl_out:
            try:
                from pylsl import StreamInfo, StreamOutlet
                info = StreamInfo("Unicorn", "EEG", N_EEG, FS, "float32", "unicorn_serial")
                chs = info.desc().append_child("channels")
                for c in CH:
                    chs.append_child("channel").append_child_value("label", c).append_child_value("unit", "microvolts")
                self.outlet = StreamOutlet(info)
                print("[source] republishing as LSL stream 'Unicorn' (type EEG)")
            except Exception as e:
                print(f"[source] (LSL republish off: {e})")

    @classmethod
    def parse(cls, fr):
        eeg = []
        for i in range(8):
            b = fr[3 + 3 * i: 6 + 3 * i]
            v = (b[0] << 16) | (b[1] << 8) | b[2]
            if v & 0x800000:
                v -= 1 << 24
            eeg.append(v * cls.UV)
        counter = int.from_bytes(fr[39:43], "little")
        battery = (fr[2] & 0x0F) * 100.0 / 15.0
        return eeg, counter, battery

    def read(self):
        from pylsl import local_clock
        self.buf += self.ser.read(max(self.FRAME, self.ser.in_waiting or 0))
        rows, counters = [], []
        while len(self.buf) >= self.FRAME:
            if self.buf[0] != 0xC0 or self.buf[1] != 0x00 or self.buf[43] != 0x0D or self.buf[44] != 0x0A:
                j = self.buf.find(b"\xC0\x00", 1)          # resync
                self.buf = self.buf[j:] if j > 0 else bytearray()
                continue
            eeg, counter, self.battery = self.parse(bytes(self.buf[:self.FRAME]))
            del self.buf[:self.FRAME]
            if self.last_counter is not None and counter != self.last_counter + 1:
                self.dropped += max(0, counter - self.last_counter - 1)
            self.last_counter = counter
            rows.append(eeg); counters.append(counter)
        if not rows:
            return np.empty((0, N_EEG)), np.empty(0), []
        # Timing from the headset's own sample counter, NOT Bluetooth arrival time (which comes in bursts).
        # offset = clock time of counter 0. Bluetooth only ever adds delay, so the smallest
        # (arrival - counter/FS) over the last 30 s is the best estimate; it also tracks slow clock drift.
        now = local_clock()
        c = np.asarray(counters, float)
        obs = now - c[-1] / FS
        self._obs.append((now, obs))
        while self._obs and now - self._obs[0][0] > 30.0:
            self._obs.popleft()
        floor = min(o for _, o in self._obs)
        self.offset = floor if self.offset is None else self.offset + 0.1 * (floor - self.offset)
        ts = self.offset + c / FS
        self.n_samples += len(rows)
        x = np.asarray(rows)
        if self.outlet:
            self.outlet.push_chunk(x.tolist(), float(ts[-1]))
        return x, ts, []

    def close(self):
        try:
            self.ser.write(self.STOP); self.ser.close()
        except Exception:
            pass
        tot = self.n_samples + self.dropped
        print(f"\n[source] stopped. dropped samples: {self.dropped} of {tot} ({100 * self.dropped / max(tot, 1):.2f}%)")


class FakeSource:
    def __init__(self, check=False):
        self.check = check
        print("[source] FAKE synthetic EEG (eyes closed every other 20 s)")
        from pylsl import local_clock
        self.k = 0; self.t0 = local_clock(); self.w0 = time.time(); self.rng = np.random.default_rng(0)

    def read(self):
        time.sleep(0.1)
        n = int((time.time() - self.w0) * FS) - self.k
        t = (self.k + np.arange(n)) / FS; self.k += n
        x = self.rng.normal(0, 8, (n, N_EEG))
        closed = ((t // 20) % 2 == 1) if not self.check else ((t > 25) & (t < 45))
        x[:, 4:] += (6 + 10 * closed[:, None]) * np.sin(2 * np.pi * 10 * t)[:, None]
        if self.check:  # blinks every 2 s in the last phase
            x[:, 0] += 150 * np.exp(-0.5 * (((t - 46) % 2 - 1) / 0.08) ** 2) * (t > 46)
        return x + 3000, self.t0 + t, []


# ----------------------------------------------------------------- processing
class Processor:
    def __init__(self):
        sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
        b, a = iirnotch(50, 30, fs=FS)
        self.sos = np.vstack([sos, tf2sos(b, a)])
        self.zi = None
        self.buf = np.zeros((WIN, N_EEG)); self.seen = 0; self.since = 0

    def push(self, x):
        if not len(x):
            return []
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        out = []
        for row in y:
            self.buf = np.roll(self.buf, -1, 0); self.buf[-1] = row
            self.seen += 1; self.since += 1
            if self.seen >= 2 * WIN and self.since >= STEP:
                self.since = 0; out.append(self.features())
        return out

    def features(self):
        w = self.buf
        f, p = welch(w.T, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2)
        df = f[1] - f[0]
        tot = p[:, (f >= 1) & (f < 30)].sum(1) * df
        feat = {"t": round(self.seen / FS, 2), "channels": CH}
        for b, (lo, hi) in BANDS.items():
            bp = p[:, (f >= lo) & (f < hi)].sum(1) * df
            feat[b] = [round(float(v), 4) for v in bp]
            feat[b + "_rel"] = [round(float(v), 4) for v in bp / np.maximum(tot, 1e-12)]
        sd, ptp = w.std(0), np.ptp(w, 0)
        q = ["flat" if s < 0.5 else "noisy" if s > 100 else "artefact" if pp > 500 else "good" for s, pp in zip(sd, ptp)]
        feat["quality"] = q
        feat["std_uv"] = [round(float(s), 1) for s in sd]
        good = np.array([v == "good" for v in q]); post = np.array([c[0] in "PO" for c in CH])
        sel = good & post if (good & post).any() else good
        a, b_ = np.array(feat["alpha"]), np.array(feat["beta"])
        feat["summary"] = {"n_good": int(good.sum()),
                           "posterior_alpha_rel": float(np.mean(np.array(feat["alpha_rel"])[sel])) if sel.any() else None,
                           "beta_alpha_ratio": float(np.mean(b_[good] / np.maximum(a[good], 1e-12))) if good.any() else None}
        return feat


# ----------------------------------------------------------------- device check
def device_check(src, out: Path):
    """Guided ~60 s test: signal amplitude, eyes-closed alpha, blink detection. Prints PASS/FAIL."""
    from scipy.signal import sosfiltfilt

    def collect(seconds, label):
        print(f"\n>>> {label}  ({seconds} s)", flush=True)
        xs, t_end = [], time.time() + seconds
        while time.time() < t_end:
            x, _, _ = src.read()
            if len(x):
                xs.append(x)
            left = int(t_end - time.time())
            print(f"\r    {left:2d} s left", end="", flush=True)
        print()
        return np.vstack(xs) if xs else np.empty((0, N_EEG))

    print("\n=== UNICORN DEVICE CHECK ===  sit still, relaxed, look at the screen")
    collect(5, "Settling, sit still")
    opn = collect(20, "EYES OPEN, look at a point on the screen, don't blink much")
    cls = collect(20, "EYES CLOSED, relax and keep them closed until you hear the beep")
    print("\a", end="")
    blk = collect(10, "EYES OPEN, now BLINK FIRMLY 5 times, about 2 s apart")

    for seg, name, sec in ((opn, "eyes open", 20), (cls, "eyes closed", 20), (blk, "blinks", 10)):
        if len(seg) < 0.8 * sec * FS:
            print(f"\n❌ NO DATA: got {len(seg)} samples during '{name}' (expected ~{sec * FS}).\n"
                  "   The headset isn't streaming. Check it's ON, close Unicorn Suite / other laptops,\n"
                  "   keep the Mac within 1 m, then run 1) again.")
            return False
    sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
    b, a = iirnotch(50, 30, fs=FS); sos = np.vstack([sos, tf2sos(b, a)])
    filt = lambda x: sosfiltfilt(sos, x - np.median(x, 0), axis=0)[int(FS):-int(FS)]
    yo, yc, yb = filt(opn), filt(cls), filt(blk)

    def alpha(y):
        f, p = welch(y.T, fs=FS, nperseg=int(2 * FS), average="median")  # robust to movement bursts
        return p[:, (f >= 8) & (f < 13)].mean(1)

    def mains_mv(x):  # 50 Hz amplitude in the RAW signal (before the notch), mV
        f, p = welch((x - x.mean(0)).T, fs=FS, nperseg=int(2 * FS), average="median")
        return np.sqrt(2 * p[:, np.argmin(np.abs(f - 50))] * (f[1] - f[0])) / 1000

    sd = np.median(yo[:len(yo) // int(FS) * int(FS)].reshape(-1, int(FS), N_EEG).std(1), 0)
    ratio = alpha(yc) / np.maximum(alpha(yo), 1e-12)
    post = [i for i, c in enumerate(CH) if c[0] in "PO"]
    fz = yb[:, 0]
    thr = max(80.0, 5 * np.median(np.abs(fz)))
    above = np.flatnonzero(fz > thr)
    blinks = int((np.diff(above) > FS * 0.4).sum() + (1 if len(above) else 0))

    mains = mains_mv(opn)
    ok_ch = [(2 <= s <= 60) and m < 5.0 for s, m in zip(sd, mains)]   # <5 mV hum: the 50 Hz notch removes it
    alpha_ok = np.median(ratio[post]) >= 1.3
    blink_ok = 3 <= blinks <= 8
    print("\n=== RESULTS ===")
    print(f"{'ch':4s} {'σ µV (eyes open)':>17s} {'50 Hz mV':>9s} {'α closed/open':>14s}  status")
    for i, c in enumerate(CH):
        st = (("OK" if mains[i] < 1.0 else "OK (some hum, filtered out)") if ok_ch[i]
              else "MAINS HUM: poor contact / reference" if mains[i] >= 5.0
              else "FLAT" if sd[i] < 2 else "NOISY: re-seat/wet electrode")
        print(f"{c:4s} {sd[i]:17.1f} {mains[i]:9.2f} {ratio[i]:14.2f}  {st}")
    if np.median(mains) >= 5.0:
        print("\n⚡ Mains hum on most channels → re-stick the REFERENCE/GROUND electrodes behind the ears,\n"
              "   unplug the Mac charger, move away from power strips/monitors.")
    print(f"\nSignal amplitude : {sum(ok_ch)}/8 channels in 2-60 µV, low hum {'PASS' if sum(ok_ch) >= 7 else 'FAIL'}")
    print(f"Eyes-closed alpha: posterior ×{np.median(ratio[post]):.2f} (need ≥1.3)        {'PASS' if alpha_ok else 'FAIL'}")
    print(f"Blink detection  : {blinks} blinks seen on Fz (asked for 5)     {'PASS' if blink_ok else 'CHECK'}")
    verdict = sum(ok_ch) >= 7 and alpha_ok
    print(f"\nOVERALL: {'✅ READY TO RECORD' if verdict else '❌ NOT READY: fix the channels marked NOISY/FLAT, then rerun'}")
    path = out / f"device_check_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["phase"] + [f"EEG {i + 1}" for i in range(N_EEG)])
        for ph, x in (("open", opn), ("closed", cls), ("blink", blk)):
            for row in x:
                w.writerow([ph] + [f"{v:.3f}" for v in row])
    print(f"Raw check data saved: {path.resolve()}")
    return verdict


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "lsl", "serial", "unicornpy"], default="auto")
    ap.add_argument("--serial", dest="serial_port", default=None, help="serial port for --source serial (default: /dev/cu.UN-*)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--name", default=None, help="file name (default unicorn_<date_time>.csv)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the ~60 s guided device test and exit")
    a = ap.parse_args()

    if a.fake:
        src = FakeSource(check=a.check)
    elif a.source == "unicornpy":
        src = UnicornPySource()
    elif a.source == "serial":
        src = SerialSource(a.serial_port)
    elif a.source == "lsl":
        try:
            src = LSLSource()
        except Exception as e:
            raise SystemExit(str(e))
    else:  # auto: serial (Mac/Linux) -> LSL -> UnicornPy (Windows)
        errs, src = [], None
        for name, make in (("Bluetooth serial", lambda: SerialSource(a.serial_port)),
                           ("LSL", lambda: LSLSource(timeout=4)), ("UnicornPy", UnicornPySource)):
            try:
                src = make(); break
            except Exception as e:
                errs.append(f"  {name}: {e}"); print(f"[source] {name} not available, trying next…")
        if src is None:
            raise SystemExit("Could not connect to the Unicorn.\n" + "\n".join(errs))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.check:
        try:
            ok = device_check(src, out)
        finally:
            if hasattr(src, "close"):
                src.close()
        raise SystemExit(0 if ok else 1)
    path = out / (a.name or f"unicorn_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(path, "w", newline=""); w = csv.writer(fh)
    w.writerow(["lsl_time"] + [f"EEG {i + 1}" for i in range(N_EEG)] + ["marker"])
    print(f"[record] writing ONE file: {path.resolve()}")

    proc, latest, clients, raw_clients = Processor(), {"f": None}, set(), set()
    import collections
    raw_q = collections.deque(maxlen=5000)   # raw samples for the live pipeline board
    stop = threading.Event()

    markers = MarkerListener()
    item_models = ItemModels()
    prediction_path = path.with_suffix(".predictions.jsonl")
    prediction_fh = open(prediction_path, "a")
    print(f"[models] item predictions: {prediction_path.resolve()}")

    def acquire():
        n, last_print, pending, n_mk = 0, time.time(), [], 0
        while not stop.is_set():
            try:
                x, ts, mks = src.read()
            except Exception as e:
                print(f"\n[source] headset disconnected / read error: {e}\n[source] stopping. Data so far is saved; rerun to continue.")
                stop.set(); break
            new = mks + markers.poll()
            item_models.push(x, ts, new)
            if item_models.updated:
                state = dict(item_models.state)
                if state["status"] != "collecting":
                    prediction_fh.write(json.dumps(state) + "\n"); prediction_fh.flush()
                item_models.updated = False
            if new:
                pending = sorted(pending + new); n_mk += len(new)
            for row, t in zip(x, ts):
                m = ""
                while pending and pending[0][0] <= t:      # attach marker to nearest following sample
                    m = (m + ";" if m else "") + pending.pop(0)[1]
                w.writerow([f"{t:.6f}"] + [f"{v:.3f}" for v in row] + [m])
            n += len(x)
            if raw_clients:
                raw_q.extend(zip(ts.tolist(), x.round(3).tolist()))
            for f in proc.push(x):
                f["model"] = dict(item_models.state)
                latest["f"] = f
            if time.time() - last_print > 1 and latest["f"]:
                f = latest["f"]; last_print = time.time(); fh.flush()
                q = " ".join(f"{c}:{'ok' if s == 'good' else s[:5]:5s}" for c, s in zip(CH, f["quality"]))
                pa = f["summary"]["posterior_alpha_rel"]
                print(f"\r{n / FS:6.0f}s  mk {n_mk:3d}  good {f['summary']['n_good']}/8  postα {pa if pa is None else round(pa, 2)}  "
                      f"σ(µV) {f['std_uv']}  {q}", end="", flush=True)
        fh.flush()

    th = threading.Thread(target=acquire, daemon=True); th.start()

    async def serve():
        import websockets

        async def handler(ws):
            clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                clients.discard(ws)
        async def raw_handler(ws):
            raw_clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                raw_clients.discard(ws)

        async def pump_raw():
            while not stop.is_set():
                await asyncio.sleep(0.05)
                if not raw_q:
                    continue
                items = [raw_q.popleft() for _ in range(len(raw_q))]
                msg = json.dumps({"type": "raw", "fs": FS, "ch": CH, "t": [i[0] for i in items], "x": [i[1] for i in items]})
                for c in list(raw_clients):
                    try:
                        await c.send(msg)
                    except Exception:
                        raw_clients.discard(c)

        async with websockets.serve(handler, "0.0.0.0", a.port), websockets.serve(raw_handler, "0.0.0.0", a.port + 1):
            print(f"[ws] features : ws://localhost:{a.port}   -> dashboard/mindcart.html?ws=ws://localhost:{a.port}")
            print(f"[ws] raw EEG  : ws://localhost:{a.port + 1} -> dashboard/pipeline.html?ws=ws://localhost:{a.port + 1}")
            asyncio.get_running_loop().create_task(pump_raw())
            last = None
            while not stop.is_set():
                await asyncio.sleep(0.2)
                f = latest["f"]
                if f is not None and f is not last:
                    last = f; msg = json.dumps(f)
                    for c in list(clients):
                        try:
                            await c.send(msg)
                        except Exception:
                            clients.discard(c)

    def keep_recording():
        while th.is_alive():
            time.sleep(0.5)

    try:
        if a.no_ws:
            keep_recording()
        else:
            try:
                asyncio.run(serve())
            except OSError as e:   # dashboard port busy (old stream still running?) -> NEVER stop recording
                print(f"\n[ws] dashboards OFF: {e}\n[ws] recording continues. Close other streams to get dashboards back.",
                      flush=True)
                keep_recording()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); th.join(timeout=2); fh.flush(); fh.close(); prediction_fh.close()
        if hasattr(src, "close"):
            src.close()
        print(f"\n[record] saved {path.resolve()}", flush=True)
        os._exit(0)   # LSL background threads can otherwise keep the process alive after Ctrl+C


if __name__ == "__main__":
    main()
