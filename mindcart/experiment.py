"""MindCart viewing task (PsychoPy). Sends LSL markers for every event.

Trial: fixation 1 s -> outfit image 4 s -> "WANT it?" 1-5 -> "NEED it?" 1-5
       -> jittered blank 0.5-0.9 s. Break screen halfway. Esc quits safely.
Impulse = high WANT + low NEED (what UnSway nudges on).

    python mindcart/experiment.py --pid P01 --items stimuli/items.csv --img-dir stimuli/norm
    python mindcart/experiment.py --pid TEST --n 5 --windowed     # 5-trial sync check
    python mindcart/experiment.py --pid OB01 --oddball            # ~4 min P300 validation (no images needed)

Oddball: blue circles (80%) and rare orange squares (20%), 200 ms each, every 0.9-1.2 s.
The participant silently counts the orange squares. Targets should give a clear P300 at Pz.

Markers (string, one per event):
  start | fix|<trial> | img|<trial>|<item_id> | rate|<trial>|<item_id>|<want>|<rt> | need|<trial>|<item_id>|<need>|<rt> | break | end
  oddball: odd|<trial>|target  /  odd|<trial>|standard
Also writes sessions/<pid>/behaviour.csv as a backup.
"""
import argparse
import csv
import random
import zlib
from pathlib import Path

from pylsl import StreamInfo, StreamOutlet, local_clock
from psychopy import core, event, visual

FIX_S, IMG_S, RATE_MAX_S = 1.0, 4.0, 6.0


def _screen_size():
    """Primary screen size in pixels. Newer PsychoPy rejects size=None even for fullscreen."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
    except Exception:
        return (1280, 800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--items", default="stimuli/items.csv")
    ap.add_argument("--img-dir", default="stimuli/norm")
    ap.add_argument("--n", type=int, default=None, help="limit number of trials (testing)")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="sessions")
    ap.add_argument("--oddball", action="store_true", help="run the P300 oddball validation instead")
    ap.add_argument("--n-oddball", type=int, default=200)
    a = ap.parse_args()
    if a.oddball:
        return run_oddball(a)

    with open(a.items) as f:
        items = list(csv.DictReader(f))
    rng = random.Random(a.seed if a.seed is not None else zlib.crc32(a.pid.encode()))  # stable per participant
    rng.shuffle(items)
    if a.n:
        items = items[:a.n]

    outlet = StreamOutlet(StreamInfo("MindCartMarkers", "Markers", 1, 0, "string", f"mindcart_{a.pid}"))
    out = Path(a.out) / a.pid
    out.mkdir(parents=True, exist_ok=True)
    beh = open(out / "behaviour.csv", "w", newline="")
    bw = csv.writer(beh)
    bw.writerow(["trial", "item_id", "onset_lsl", "want", "want_rt", "need", "need_rt"])

    win = visual.Window(size=(1280, 800) if a.windowed else _screen_size(), fullscr=not a.windowed,
                        color=(0, 0, 0), units="height")  # mid-grey, matches image background
    fix = visual.TextStim(win, "+", height=0.08, color="white")
    msg = visual.TextStim(win, "", height=0.04, color="white", wrapWidth=1.2)
    q_want = visual.TextStim(win, "How much do you WANT it?\n\n1   2   3   4   5\n\n"
                                  "(1 = not at all, 5 = I'd buy it now)", height=0.045, color="white")
    q_need = visual.TextStim(win, "How much do you NEED it?\n\n1   2   3   4   5\n\n"
                                  "(1 = not at all, 5 = I really need it)", height=0.045, color="white")
    img = visual.ImageStim(win, size=(0.8, 0.8))

    def mark(s):
        outlet.push_sample([s], local_clock())

    def show_msg(text):
        msg.text = text
        msg.draw(); win.flip()
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            quit_()

    def quit_():
        mark("end"); beh.close(); win.close(); core.quit()

    show_msg("Look at each clothing item while keeping still and relaxed.\n"
             "Try not to blink while the item is on screen.\n\n"
             "Then answer two quick questions with keys 1-5:\nhow much you WANT it, and how much you NEED it.\n"
             "Go with your gut, there are no right answers.\n\nPress SPACE to start.")
    core.wait(1.0)  # give the recorder a moment
    mark("start")
    half = len(items) // 2
    clock = core.Clock()
    for t, it in enumerate(items):
        if t == half and len(items) > 10:
            mark("break")
            show_msg("Short break. Blink, stretch, relax.\n\nPress SPACE to continue.")
        img.image = str(Path(a.img_dir) / it.get("norm_file", it["file"]))
        # fixation
        fix.draw(); win.callOnFlip(mark, f"fix|{t}"); win.flip(); core.wait(FIX_S)
        # product
        img.draw()
        onset = {}
        win.callOnFlip(lambda: onset.setdefault("t", local_clock()))
        win.callOnFlip(mark, f"img|{t}|{it['item_id']}")
        win.flip(); core.wait(IMG_S)
        # ratings: WANT then NEED
        def ask(stim):
            stim.draw(); win.flip(); clock.reset(); event.clearEvents()
            keys = event.waitKeys(maxWait=RATE_MAX_S, keyList=list("12345") + ["escape"], timeStamped=clock)
            if keys and keys[0][0] == "escape":
                quit_()
            return (int(keys[0][0]), round(keys[0][1], 3)) if keys else ("", "")
        w, wrt = ask(q_want)
        mark(f"rate|{t}|{it['item_id']}|{w}|{wrt}")
        win.flip(); core.wait(0.2)
        n, nrt = ask(q_need)
        mark(f"need|{t}|{it['item_id']}|{n}|{nrt}")
        bw.writerow([t, it["item_id"], f"{onset.get('t', 0):.6f}", w, wrt, n, nrt]); beh.flush()
        win.flip(); core.wait(rng.uniform(0.5, 0.9))
    show_msg("Done, thank you!\n\nPress SPACE to finish.")
    quit_()


def run_oddball(a):
    rng = random.Random(a.seed if a.seed is not None else zlib.crc32(a.pid.encode()))
    # 20% targets, never two targets in a row, first 3 are standards
    kinds = ["standard"] * 3
    while len(kinds) < a.n_oddball:
        kinds.append("target" if (rng.random() < 0.25 and kinds[-1] != "target") else "standard")
    outlet = StreamOutlet(StreamInfo("MindCartMarkers", "Markers", 1, 0, "string", f"mindcart_{a.pid}"))
    win = visual.Window(size=(1280, 800) if a.windowed else _screen_size(), fullscr=not a.windowed, color=(0, 0, 0), units="height")
    msg = visual.TextStim(win, "", height=0.04, color="white", wrapWidth=1.2)
    fix = visual.TextStim(win, "+", height=0.06, color="white")
    std = visual.Circle(win, radius=0.08, fillColor=(-0.6, -0.2, 0.9), lineColor=None)
    tgt = visual.Rect(win, width=0.16, height=0.16, fillColor=(1.0, 0.3, -0.6), lineColor=None)
    mark = lambda s: outlet.push_sample([s], local_clock())

    def wait_space(text):
        msg.text = text; msg.draw(); win.flip()
        if "escape" in event.waitKeys(keyList=["space", "escape"]):
            mark("end"); win.close(); core.quit()

    wait_space("Keep your eyes on the centre and stay still.\n\nCOUNT the ORANGE SQUARES silently.\n"
               "Ignore the blue circles. Try not to blink while shapes are on screen.\n\nPress SPACE to start.")
    core.wait(1.0); mark("start")
    for k, kind in enumerate(kinds):
        (tgt if kind == "target" else std).draw(); fix.draw()
        win.callOnFlip(mark, f"odd|{k}|{kind}"); win.flip(); core.wait(0.2)
        fix.draw(); win.flip(); core.wait(rng.uniform(0.7, 1.0))
        if event.getKeys(["escape"]):
            break
    n_t = kinds.count("target")
    wait_space(f"Done! How many orange squares did you count?\n(Answer: {n_t})\n\nPress SPACE.")
    mark("end"); win.close(); core.quit()


if __name__ == "__main__":
    main()
