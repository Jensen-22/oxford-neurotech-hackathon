/* live-eeg.js — feeds the "second thoughts" demo with live Unicorn data.
 *
 *   Live EEG  + Brain activity bands  ← feature frames (1/s) on ws://localhost:8765  (same source as dashboard/mindcart.html)
 *   Band power (recent 10 s)          ← the per-second "Output to the app" frame of dashboard/pipeline.html, recomputed
 *                                        here from the raw stream on ws://localhost:8766 (same filters / Welch / bands)
 *   Live EEG traces                   ← raw samples on ws://localhost:8766 (1–40 Hz + 50 Hz notch, as pipeline.html step 2)
 *
 * Nothing in app.js is touched: this watches #app and only rewrites the three cards above.
 * URL overrides: ?ws=ws://host:8765  ?raw=ws://host:8766
 */
(function () {
  const q = new URLSearchParams(location.search);
  const FEAT_URL = q.get('ws') || 'ws://localhost:8765';
  const RAW_URL = q.get('raw') || (() => { try { const u = new URL(FEAT_URL); u.port = String((+u.port || 8765) + 1); return u.toString().replace(/\/$/, ''); } catch (e) { return 'ws://localhost:8766'; } })();

  // ---- constants (mirror pipeline.html) ----
  const FS = 250, CH = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'], NCH = 8, BUF = FS * 10, WIN = 512;
  const BANDS = { delta: [1, 4], theta: [4, 8], alpha: [8, 13], beta: [13, 30] };
  const POST = [4, 5, 6, 7];
  const SHOW = [0, 1, 2, 3];           // channels drawn in the "Live EEG (4 channels)" card: Fz C3 Cz C4
  const HIST_S = 10;                   // "recent 10 s" shown on the chart
  const KEEP_S = 60;                   // seconds of history kept, so the freeze can look back for the real peak

  // ---- ring buffer ----
  class Ring { constructor(n) { this.a = CH.map(() => new Float32Array(n)); this.n = n; this.i = 0; this.count = 0; }
    push(v) { for (let c = 0; c < NCH; c++) this.a[c][this.i] = v[c]; this.i = (this.i + 1) % this.n; this.count++; }
    last(c, k) { const o = new Float32Array(k); for (let j = 0; j < k; j++) o[j] = this.a[c][(this.i - k + j + this.n * 2) % this.n]; return o; } }

  // ---- filters (RBJ biquads, DF2T) ----
  function biquad(type, f0, Q) { const w = 2 * Math.PI * f0 / FS, c = Math.cos(w), s = Math.sin(w), al = s / (2 * Q); let b0, b1, b2; const a0 = 1 + al, a1 = -2 * c, a2 = 1 - al;
    if (type === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; } else if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = (1 - c) / 2; } else { b0 = 1; b1 = -2 * c; b2 = 1; }
    return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: 0, z2: 0 }; }
  const makeChain = () => [biquad('hp', 1, .5412), biquad('hp', 1, 1.3066), biquad('lp', 40, .5412), biquad('lp', 40, 1.3066), biquad('notch', 50, 30)];
  function runChain(ch, x) { for (const f of ch) { const y = f.b0 * x + f.z1; f.z1 = f.b1 * x - f.a1 * y + f.z2; f.z2 = f.b2 * x - f.a2 * y; x = y; } return x; }

  // ---- FFT + Welch ----
  function fft(re, im) { const n = re.length; for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a); for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) {
      const ur = re[i + k], ui = im[i + k], vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci, vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr; re[i + k] = ur + vr; im[i + k] = ui + vi; re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
      const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } } }
  const SEG = 256, HANN = Float32Array.from({ length: SEG }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / (SEG - 1))), U = HANN.reduce((s, v) => s + v * v, 0);
  function welch(x) { const p = new Float64Array(SEG / 2 + 1); let k = 0; for (let s = 0; s + SEG <= x.length; s += SEG / 2, k++) {
    let m = 0; for (let i = 0; i < SEG; i++) m += x[s + i]; m /= SEG; const re = new Float64Array(SEG), im = new Float64Array(SEG); for (let i = 0; i < SEG; i++) re[i] = (x[s + i] - m) * HANN[i]; fft(re, im);
    for (let f = 0; f <= SEG / 2; f++) p[f] += (re[f] * re[f] + im[f] * im[f]) / (FS * U) * (f && f < SEG / 2 ? 2 : 1); } for (let f = 0; f < p.length; f++) p[f] /= k; return p; }
  const DF = FS / SEG, FREQ = Array.from({ length: SEG / 2 + 1 }, (_, i) => i * DF);
  const bandSum = (p, lo, hi) => { let s = 0; for (let i = 0; i < p.length; i++) if (FREQ[i] >= lo && FREQ[i] < hi) s += p[i] * DF; return s; };

  // ---- state ----
  const S = { feat: null, featAt: 0, hist: [], scoreHist: [], deltaHist: [], rawLive: false, featLive: false, bp: null, frozen: null, live: null, score: null };
  let FILT = new Ring(BUF), filt = CH.map(makeChain), dc = null, sinceFeat = 0, nSamples = 0;

  function ingest(v) {
    if (!dc) dc = v.slice();
    FILT.push(v.map((x, c) => runChain(filt[c], x - dc[c]))); nSamples++;
    if (++sinceFeat >= FS && FILT.count >= 2 * WIN) { sinceFeat = 0; computeFrame(); }
  }
  // == pipeline.html step 3–6: the exact "Output to the app" frame, once per second ==
  function computeFrame() {
    const win = CH.map((_, c) => FILT.last(c, WIN));
    const qual = win.map(w => { let m = 0; for (const x of w) m += x; m /= w.length; let s = 0, mn = 1e9, mx = -1e9; for (const x of w) { s += (x - m) ** 2; mn = Math.min(mn, x); mx = Math.max(mx, x); } const sd = Math.sqrt(s / w.length), pp = mx - mn;
      return sd < .5 ? 'flat' : sd > 100 ? 'noisy' : pp > 150 ? 'artefact' : 'good'; });
    const P = win.map(welch);
    const bp = P.map(p => { const tot = bandSum(p, 1, 30) || 1e-12; const o = {}; for (const [b, [lo, hi]] of Object.entries(BANDS)) { o[b] = bandSum(p, lo, hi); o[b + '_rel'] = o[b] / tot; } return o; });
    const good = qual.map(s => s === 'good'), sel = POST.filter(i => good[i]); const use = sel.length ? sel : POST;
    const pa = use.reduce((s, i) => s + bp[i].alpha_rel, 0) / use.length, th = bp[0].theta_rel;
    const gi = CH.map((_, i) => i).filter(i => good[i]), ba = (gi.length ? gi : CH.map((_, i) => i)).reduce((s, i) => s + bp[i].beta / Math.max(bp[i].alpha, 1e-12), 0) / (gi.length || NCH);
    const eng = (gi.length ? gi : CH.map((_, i) => i)).reduce((s, i) => s + bp[i].beta / Math.max(bp[i].alpha + bp[i].theta, 1e-12), 0) / (gi.length || NCH); // Pope engagement index β/(α+θ)
    S.bp = bp; S.quality = qual;
    S.hist.push({ t: nSamples / FS, pa, th, ba }); while (S.hist.length > KEEP_S + 1) S.hist.shift();
    updateScore({ pa, th, ba, eng });
    if (S.pendingFreeze) freeze();
    drawBandPower();
    if (!S.featLive) drawBands();   // fallback if the feature socket is down
  }

  // ---- "Possible impulse cue" score, from the same per-second frame ----
  // Every option is relative to a slowly adapting personal baseline (EMA, as in mindcart.html), then squashed to 0–100 %.
  //   ?score=mindcart   (default) mindcart.html's formula:  z = 1.5·Δθ(Fz) − 1.5·Δα(posterior) + 0.5·Δ(β/α)
  //   ?score=alpha      posterior alpha suppression:        z = 3·(−Δα)            (less idle alpha → more engaged with the item)
  //   ?score=theta      frontal theta rise:                 z = 3·Δθ(Fz)           (conflict / craving marker)
  //   ?score=engagement Pope engagement index β/(α+θ):      z = 2·ΔE
  //   Δx = (x − baseline) / baseline
  const SCORE_MODE = q.get('score') || 'mindcart';
  let base = null;
  function updateScore(f) {
    if (!base) base = { ...f };
    const d = {}; for (const k in f) { d[k] = (f[k] - base[k]) / (base[k] + 1e-6); base[k] = .95 * base[k] + .05 * f[k]; }
    const z = SCORE_MODE === 'alpha' ? -3 * d.pa : SCORE_MODE === 'theta' ? 3 * d.th : SCORE_MODE === 'engagement' ? 2 * d.eng
      : 1.5 * d.th - 1.5 * d.pa + .5 * d.ba;
    S.live = 1 / (1 + Math.exp(-z));
    S.scoreHist.push(S.live); S.deltaHist.push(d); while (S.scoreHist.length > S.hist.length) { S.scoreHist.shift(); S.deltaHist.shift(); }
    if (!S.frozen) S.score = S.live;
    drawScore();
  }
  function drawScore() { const el = document.querySelector('.impulse .pct'); if (el && S.score != null) el.textContent = Math.round(100 * S.score) + '%'; }

  // ---- freeze on "Add to bag": snapshot the last 10 s and pin the score at its peak, so the pop-up can show *why* it fired ----
  function freeze() {
    if (S.hist.length < 2) { S.pendingFreeze = true; return; }   // no data yet: freeze as soon as the first frames arrive
    S.pendingFreeze = false;
    // look back over the whole kept history (60 s) for the strongest impulse index, then show the 10 s window around it
    let pk = 0; S.scoreHist.forEach((v, i) => { if (v > S.scoreHist[pk]) pk = i; });
    const tPk = S.hist[pk].t, tEnd = Math.min(S.hist[S.hist.length - 1].t, tPk + 2), tStart = tEnd - HIST_S;
    let idx = S.hist.map((p, i) => i).filter(i => S.hist[i].t > tStart && S.hist[i].t <= tEnd);
    if (idx.length < 2) idx = S.hist.map((p, i) => i).slice(-HIST_S - 1);
    S.frozen = { hist: idx.map(i => S.hist[i]), scores: idx.map(i => S.scoreHist[i]), peak: idx.indexOf(pk), delta: S.deltaHist[pk], at: Date.now() };
    if (S.frozen.peak < 0) S.frozen.peak = S.frozen.scores.indexOf(Math.max(...S.frozen.scores));
    S.score = S.scoreHist[pk]; drawScore(); drawBandPower(); drawCueNote();
  }
  function unfreeze() { S.pendingFreeze = false; if (S.frozen) { S.frozen = null; S.score = S.live; } }
  // one line of evidence under the chart: which bands moved at the peak, vs the personal baseline
  const NAMES = { th: 'Fz θ', pa: 'posterior α', ba: 'β/α' };
  function drawCueNote() {
    const card = document.querySelector('.chart'); if (!card) return;
    const em = card.parentElement.querySelector('h3 em'); let note = card.parentElement.querySelector('.cue-note');
    if (!S.frozen) { if (em) em.textContent = '(recent 10 s)'; if (note) note.remove(); return; }
    if (em) em.textContent = '(10 s before the cue)';
    if (!note) { note = document.createElement('p'); note.className = 'cue-note'; note.style.cssText = 'margin-top:8px;text-align:center'; card.parentElement.appendChild(note); }
    const d = S.frozen.delta || {}; const parts = ['th', 'pa', 'ba'].filter(k => Math.abs(d[k] || 0) >= .1).sort((a, b) => Math.abs(d[b]) - Math.abs(d[a]))
      .map(k => `${NAMES[k]} ${d[k] > 0 ? '+' : '−'}${Math.round(Math.abs(d[k]) * 100)}%`);
    note.textContent = parts.length ? `At the cue: ${parts.join(' · ')} vs your baseline` : 'At the cue: no band moved more than 10% from your baseline';
  }
  document.addEventListener('click', e => { if (e.target.closest('#add-bag')) freeze(); }, true);

  // ---- sockets ----
  function openRaw() { const ws = new WebSocket(RAW_URL);
    ws.onmessage = e => { const m = JSON.parse(e.data); if (m.type !== 'raw' || !m.x) return; S.rawLive = true; for (const v of m.x) ingest(v); };
    ws.onclose = () => { S.rawLive = false; setTimeout(openRaw, 2000); }; ws.onerror = () => ws.close(); }
  function openFeat() { const ws = new WebSocket(FEAT_URL);
    ws.onmessage = e => { const f = JSON.parse(e.data); if (!f.alpha) return; S.feat = f; S.featAt = performance.now(); S.featLive = true; drawBands(); };
    ws.onclose = () => { S.featLive = false; setTimeout(openFeat, 2000); }; ws.onerror = () => ws.close(); }

  // ---- drawing: Live EEG (4 channels) ----
  const N_SHOW = FS * 2; // 2 s per trace
  function drawEEG() {
    const traces = document.querySelectorAll('.trace');
    if (!traces.length || FILT.count < 10) return;
    traces.forEach((el, k) => { const c = SHOW[k]; if (c == null) return;
      const a = FILT.last(c, Math.min(N_SHOW, FILT.count)); let s = 0; for (const v of a) s += v * v; const sd = Math.sqrt(s / a.length) || 1, sc = Math.max(8, 2.5 * sd);
      const n = a.length, step = Math.max(1, Math.floor(n / 250)); let d = '';
      for (let i = 0; i < n; i += step) { const x = 250 * i / (n - 1), y = 12 - Math.max(-1, Math.min(1, a[i] / sc)) * 11; d += (d ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1); }
      const p = el.querySelector('path'); if (p) p.setAttribute('d', d); });
  }

  // ---- drawing: Brain activity bands ----
  const fmtUV = v => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  function meanGood(arr, qual) { const g = arr.filter((_, i) => !qual || qual[i] === 'good'); const u = g.length ? g : arr; return u.reduce((s, v) => s + v, 0) / u.length; }
  function drawBands() {
    const rows = document.querySelectorAll('.band'); if (rows.length < 3) return;
    let th, al, be;
    if (S.featLive && S.feat) { const f = S.feat; th = meanGood(f.theta, f.quality); al = meanGood(f.alpha, f.quality); be = meanGood(f.beta, f.quality); }
    else if (S.bp) { th = meanGood(S.bp.map(b => b.theta), S.quality); al = meanGood(S.bp.map(b => b.alpha), S.quality); be = meanGood(S.bp.map(b => b.beta), S.quality); }
    else return;
    const tot = th + al + be || 1;
    [[rows[0], th], [rows[1], al], [rows[2], be]].forEach(([row, v]) => {
      row.querySelector('.bar').style.setProperty('--w', Math.round(100 * v / tot) + '%');
      row.querySelector(':scope > span:last-child').textContent = fmtUV(v) + ' µV²';
    });
  }

  // ---- drawing: Band power (recent 10 s), same three series as pipeline.html step 6 ----
  function smooth(pts) { if (pts.length < 2) return ''; let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) { const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6], c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += `C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`; } return d; }
  function drawBandPower() {
    const svg = document.querySelector('.chart svg'); if (!svg) return;
    const h = S.frozen ? S.frozen.hist : S.hist.slice(-HIST_S - 1); if (h.length < 2) return;
    const paths = svg.querySelectorAll('path'); if (paths.length < 3) return;  // order in markup: alpha (green), theta (blue), beta (purple)
    const t1 = h[h.length - 1].t, X = t => 300 * (1 - (t1 - t) / HIST_S), Y = v => 76 - Math.max(0, Math.min(1, v)) * 70;
    const series = [['pa', 1], ['th', 1], ['ba', .25]];  // β/α ÷ 4, as on the pipeline board
    series.forEach(([k, s], i) => paths[i].setAttribute('d', smooth(h.map(p => [X(p.t), Y(p[k] * s)]))));
    // peak marker (frozen only): dashed line + dot where the impulse index peaked
    let mk = svg.querySelector('#peak');
    if (S.frozen) {
      if (!mk) { mk = document.createElementNS('http://www.w3.org/2000/svg', 'g'); mk.id = 'peak'; mk.innerHTML = '<line stroke="#d9541a" stroke-width="1" stroke-dasharray="3 3"/><circle r="3.5" fill="#d9541a"/>'; svg.appendChild(mk); }
      const x = X(h[S.frozen.peak].t), yv = Y(S.frozen.scores[S.frozen.peak]);
      const ln = mk.querySelector('line'); ln.setAttribute('x1', x); ln.setAttribute('x2', x); ln.setAttribute('y1', 4); ln.setAttribute('y2', 78);
      const c = mk.querySelector('circle'); c.setAttribute('cx', x); c.setAttribute('cy', yv);
    } else if (mk) mk.remove();
  }

  // ---- mount: re-apply after every render of #app ----
  const COL = { alpha: '#6fae74', theta: '#6aa8dd', ratio: '#d2651f' };
  // UnSway brand in the panel header (replaces the "second thoughts" orb + text)
  const BRAND = '<img src="/images/unsway-mark.png" alt="UnSway" style="height:46px;width:auto;flex:none;display:block">'
    + '<span style="display:block;line-height:1;color:#213927;font-weight:600;font-size:26px;letter-spacing:-.6px">UnSway'
    + '<small style="display:block;margin-top:4px;font-size:10.5px;font-weight:400;font-style:italic;letter-spacing:.4px;color:#5a6a5c">Beyond the impulse.</small></span>';
  function mount() {
    document.querySelectorAll('.brand').forEach(b => { if (!b.dataset.unsway) { b.dataset.unsway = '1'; b.innerHTML = BRAND; } });
    const small = document.querySelector('.sub h3 small'); if (small && small.textContent.trim() === '256 Hz') small.textContent = FS + ' Hz';
    document.querySelectorAll('.trace b').forEach((b, k) => { if (SHOW[k] != null) b.textContent = CH[SHOW[k]]; });
    // Band power card: legend + colours as on the pipeline board (step 6)
    const legend = document.querySelector('.chart + .legend');
    if (legend) legend.innerHTML = `<span><i style="background:${COL.alpha}"></i>posterior α (rel)</span><span><i style="background:${COL.theta}"></i>Fz θ (rel)</span><span><i style="background:${COL.ratio}"></i>β/α ÷ 4</span>`;
    const paths = document.querySelectorAll('.chart svg path');
    if (paths.length >= 3) { paths[0].setAttribute('stroke', COL.alpha); paths[1].setAttribute('stroke', COL.theta); paths[2].setAttribute('stroke', COL.ratio); }
    const note = document.querySelector('.impulse .note'); if (note) note.remove();
    if (document.querySelector('#add-bag') && !document.querySelector('.card')) unfreeze();   // back on the product page → live again
    drawBands(); drawBandPower(); drawScore(); drawCueNote();
  }
  const app = document.querySelector('#app');
  new MutationObserver(muts => { if (muts.some(m => m.type === 'childList' && m.target === app)) mount(); }).observe(app, { childList: true });
  (function loop() { drawEEG(); requestAnimationFrame(loop); })();
  openRaw(); openFeat(); mount();
  window.LIVE = S;
})();
