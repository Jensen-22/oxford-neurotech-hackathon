#!/bin/bash
# UnSway one-click launcher (macOS). Double-click in Finder.
# First time: right-click → Open (macOS asks because it's a downloaded script), and allow
# Terminal to control "Google Chrome" / "Terminal" when macOS asks.
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1
PY=python3
G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; B='\033[1m'; N='\033[0m'

say() { printf "${B}%s${N}\n" "$*"; }
ok()  { printf "${G}✔ %s${N}\n" "$*"; }
warn(){ printf "${Y}! %s${N}\n" "$*"; }
err() { printf "${R}✘ %s${N}\n" "$*"; }

setup() {
  if [ ! -d .venv ]; then
    say "First-time setup (2-5 min)…"
    $PY -m venv .venv || { err "python3 not found. Install Python 3.10 from python.org"; exit 1; }
  fi
  source .venv/bin/activate
  if ! python -c "import numpy, scipy, pandas, PIL, sklearn, matplotlib, pylsl, websockets, serial" 2>/dev/null; then
    pip install -q --upgrade pip
    pip install -q numpy scipy pandas pillow scikit-learn matplotlib pylsl websockets pyserial || { err "pip install failed"; exit 1; }
  fi
  if ! python -c "import psychopy" 2>/dev/null; then
    say "Installing PsychoPy (the experiment window)…"
    pip install -q psychopy || warn "PsychoPy failed to install. Send Claude the error; the browser version of the task can replace it."
  fi
  ok "Python environment ready"
}

new_terminal() {   # run a command in a NEW Terminal window
  osascript -e "tell application \"Terminal\" to do script \"cd '$DIR' && source .venv/bin/activate && $1\"" >/dev/null
}

open_url() {       # keep ?ws=… (macOS 'open' drops query strings on file:// URLs)
  osascript -e "tell application \"Google Chrome\" to open location \"$1\"" -e 'tell application "Google Chrome" to activate' >/dev/null 2>&1 || open "$1"
}

wait_port() {
  for i in $(seq 1 40); do nc -z localhost "$1" 2>/dev/null && return 0; sleep 0.5; done
  return 1
}

kill_old_streams() {
  pkill -INT -f "stream_unicorn.py" 2>/dev/null
  for i in $(seq 1 10); do pgrep -f "stream_unicorn.py" >/dev/null || return 0; sleep 0.5; done
  pkill -9 -f "stream_unicorn.py" 2>/dev/null; sleep 0.5
}

start_stream() {   # $1 = recording file name
  kill_old_streams
  say "Starting headset stream → recordings/$1"
  new_terminal "python3 stream_unicorn.py --name $1 ${UNSWAY_FAKE:+--fake}"
  # success = the recording file exists AND keeps growing (not just an open port)
  local f="recordings/$1" s1=0 s2=0
  for i in $(seq 1 60); do
    if [ -f "$f" ]; then s1=$(wc -c < "$f"); sleep 1; s2=$(wc -c < "$f"); [ "$s2" -gt "$s1" ] && [ "$s2" -gt 5000 ] && break; else sleep 0.5; fi
  done
  if [ -f "$f" ] && [ "$s2" -gt "$s1" ]; then ok "Headset streaming and recording"
  else err "No data arriving. Check the new Terminal window (headset on? paired? battery?)"; return 1; fi
  wait_port 8766 || warn "Dashboards not available (recording still OK)"
}

open_dashboards() {
  open_url "file://$DIR/dashboard/pipeline.html?ws=ws://localhost:8766"
  sleep 0.5
  open_url "file://$DIR/dashboard/mindcart.html?ws=ws://localhost:8765"
  sleep 0.5
  open_url "file://$DIR/dashboard/brainwaves.html?ws=ws://localhost:8766"
  ok "Dashboards open in Chrome (pipeline board + UnSway)"
}

stop_stream() {
  sleep 2
  pgrep -f "stream_unicorn.py" >/dev/null || return 0
  kill_old_streams; ok "Recording stopped and saved"
}

ask_windowed() {
  read -r -p "Is the experiment on a SECOND screen (external monitor)? [y/N] " a
  [[ "$a" =~ ^[Yy] ]] && echo "" || echo "--windowed"
}

setup
while true; do
  echo
  say "════════ UnSway ════════"
  echo " 1) Headset check (1 min: eyes open / closed / blinks)"
  echo " 2) Live dashboards only (stream + open dashboards)"
  echo " 3) ODDBALL pilot (4 min): record + dashboards + task + PASS/FAIL"
  echo " 4) PARTICIPANT session (~12 min): record + dashboards + 72 outfits + ERP plot"
  echo " 5) Analyse ALL participants (features → ERP → models)"
  echo " 6) Stop everything"
  echo " q) Quit"
  read -r -p "Choose: " c
  case "$c" in
    1) python3 stream_unicorn.py --check ;;
    2) start_stream "live_$(date +%H%M%S).csv" && open_dashboards
       say "Streaming. Press ENTER here to stop."; read -r; stop_stream ;;
    3) read -r -p "Oddball ID [OB01]: " id; id=${id:-OB01}
       start_stream "$id.csv" || continue
       open_dashboards; W=$(ask_windowed)
       say "Participant: COUNT the ORANGE SQUARES. Stay still, avoid blinking."
       python3 mindcart/experiment.py --pid "$id" --oddball $W $UNSWAY_TASK_ARGS
       stop_stream
       python3 -m mindcart.erp "recordings/$id.csv" --oddball && open "recordings/$id/oddball_p300.png" ;;
    4) read -r -p "Participant ID (e.g. P01): " id; [ -z "$id" ] && continue
       [ -f "recordings/$id.csv" ] && { warn "recordings/$id.csv exists, use another ID"; continue; }
       start_stream "$id.csv" || continue
       open_dashboards; W=$(ask_windowed)
       python3 mindcart/experiment.py --pid "$id" --items stimuli/items.csv --img-dir stimuli/norm $W $UNSWAY_TASK_ARGS
       stop_stream
       python3 -m mindcart.features "recordings/$id.csv" && python3 -m mindcart.erp "recordings/$id.csv" \
         && open "recordings/$id/erp_want.png" ;;
    5) files=$(ls recordings/P*.csv 2>/dev/null | grep -v '/live_')
       [ -z "$files" ] && { warn "No recordings/P*.csv yet"; continue; }
       echo "Using: $files"
       python3 -m mindcart.features $files && python3 -m mindcart.erp $files \
         && python3 -m mindcart.models $files --items stimuli/items.csv && open recordings/results.png ;;
    6) stop_stream ;;
    q|Q) stop_stream; exit 0 ;;
  esac
done
