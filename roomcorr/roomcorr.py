#!/usr/bin/env python3
"""
roomcorr.py - standalone room-correction toolkit (measure -> design FIR -> audition)

Ideas borrowed from Dirac Live:
  * multi-position measurement (sweep + deconvolution, optional loopback reference)
  * frequency-dependent windowing (long windows at LF, short at HF)
  * magnitude from the power average of all positions
  * mixed-phase FIR: minimum-phase magnitude correction + low-frequency
    excess-phase correction, weighted by inter-position coherence
  * bounded correction (max boost / max cut / band limits), user target curve
  * MIMO: least-squares multi-source / multi-microphone solver (e.g. 2-4 subs)

Install:   pip install numpy scipy sounddevice      (matplotlib optional: --plot)

Workflow:
  python roomcorr.py devices
  python roomcorr.py measure --channel L --pos 1 --out-ch 1 --in-ch 1 --loop-ch 2
      (repeat for pos 2..9, then for channel R; channel names must not end in _<number>)
  python roomcorr.py design --meas meas --out filters --taps 16384 --plot
  python roomcorr.py apply in.wav out.wav --left filters/L.wav --right filters/R.wav
  python roomcorr.py mimo --meas meas --channels subA,subB --out filters --fmax 150
  python roomcorr.py selftest          # synthetic room, no audio hardware needed

Output filters are mono float32 WAV files, usable by any FIR/convolution engine.
The filters contain a modelling delay (--delay-ms) so the excess-phase part is causal.
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.io import wavfile

TINY = 1e-12


# ------------------------------------------------------------------ helpers
def log_sweep(fs, dur, f1, f2, fade=0.02):
    n = int(fs * dur)
    t = np.arange(n) / fs
    r = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * dur / r * (np.exp(t * r / dur) - 1))
    m = int(fade * fs)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    x[:m] *= ramp
    x[-m:] *= ramp[::-1]
    return x


def deconvolve(rec, sweep):
    """Regularised spectral division -> impulse response (index 0 = start of sweep)."""
    n = next_fast_len(len(rec) + len(sweep))
    X, Y = rfft(sweep, n), rfft(rec, n)
    p = np.abs(X) ** 2
    return irfft(Y * np.conj(X) / (p + 1e-6 * p.max()), n)


def crop(x, start, n):
    """x[start:start+n], zero padded on both sides."""
    out = np.zeros(n)
    lo = max(start, 0)
    seg = x[lo:start + n]
    out[lo - start:lo - start + len(seg)] = seg
    return out


def extract_ir(full, ref_full, ir_len, pre):
    src = ref_full if ref_full is not None else full
    t0 = int(np.argmax(np.abs(src[:len(src) // 2])))
    return crop(full, t0 - pre, ir_len)


def oct_smooth(x, width):
    """Fractional-octave moving average on a linear-frequency grid (width in octaves)."""
    n = len(x)
    k = np.arange(n)
    w = np.broadcast_to(np.asarray(width, float), (n,))
    c = np.concatenate(([0.0], np.cumsum(x)))
    lo = np.clip(np.floor(k * 2 ** (-w / 2)).astype(int), 0, n - 1)
    hi = np.clip(np.ceil(k * 2 ** (w / 2)).astype(int) + 1, lo + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def smooth_width(f):
    return np.interp(f, [300, 1500, 6000], [1 / 12, 1 / 6, 1 / 2])


def band_weight(f, f1, f2, oct_lo=1.0, oct_hi=1.0):
    """1 inside [f1,f2], cosine fade to 0 over oct_lo octaves below / oct_hi above."""
    lf = np.log2(np.maximum(f, 1e-3))
    lo = np.clip((lf - np.log2(f1)) / oct_lo + 1, 0, 1)
    hi = np.clip(1 - (lf - np.log2(f2)) / oct_hi, 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi))


def minphase_phase(logmag, nfft):
    """Phase of the minimum-phase response whose log-magnitude is `logmag` (rfft bins)."""
    c = irfft(logmag, nfft)
    fold = np.zeros(nfft)
    fold[0] = c[0]
    fold[1:nfft // 2] = 2 * c[1:nfft // 2]
    fold[nfft // 2] = c[nfft // 2]
    return rfft(fold).imag


def wmedian(x, w):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return x[i][min(np.searchsorted(c, 0.5 * c[-1]), len(x) - 1)]


def nfft_for(taps):
    return 1 << int(np.ceil(np.log2(max(131072, 8 * taps))))


def load_curve(path):
    fx, dbx = [], []
    for line in open(path):
        p = line.replace(",", " ").replace("\t", " ").split()
        if len(p) >= 2:
            try:
                fx.append(float(p[0]))
                dbx.append(float(p[1]))
            except ValueError:
                pass
    o = np.argsort(fx)
    return np.array(fx)[o], np.array(dbx)[o]


def curve_at(f, curve):
    fx, dbx = curve
    return np.interp(np.log2(np.maximum(f, 1e-3)), np.log2(fx), dbx)


def target_db(f, a):
    t = np.zeros_like(f)
    if getattr(a, "target_file", None):
        t += curve_at(f, load_curve(a.target_file))
    pts = getattr(a, "target_points", None)
    if pts:
        pts = sorted(pts)
        t += curve_at(f, (np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)))
    if getattr(a, "bass", 0):
        t += a.bass / (1 + (f / a.bass_f) ** 2)
    if getattr(a, "tilt", 0):
        t += a.tilt * np.log2(np.maximum(f, 20) / 1000.0)
    return t


def read_wav(path):
    fs, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x / float(np.iinfo(x.dtype).max)
    elif x.dtype.kind == "u":
        x = (x.astype(float) - 128) / 128
    return fs, x.astype(float)


def save_meas(d, ch, pos, ir, fs, has_ref):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"{ch}_{pos:02d}.npz"), ir=ir, fs=fs, ref=has_ref)


def load_meas(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p)
        ch, pos = os.path.basename(p)[:-4].rsplit("_", 1)
        out.setdefault(ch, {})[int(pos)] = (z["ir"].astype(float), int(z["fs"]))
    if not out:
        sys.exit(f"no *.npz measurements in {d}")
    fss = {v[1] for c in out.values() for v in c.values()}
    if len(fss) != 1:
        sys.exit(f"mixed sample rates in {d}: {fss}")
    return out


# ------------------------------------------------------------------ measure
def cmd_devices(a):
    import sounddevice as sd
    print(sd.query_devices())


def cmd_measure(a):
    import sounddevice as sd
    fs = a.fs
    f2 = min(a.f2, 0.45 * fs)
    gain = 10 ** (a.level / 20)
    sweep = log_sweep(fs, a.dur, a.f1, f2) * gain
    play = np.concatenate([np.zeros(int(0.3 * fs)), sweep, np.zeros(int(a.tail * fs))])
    in_map = [a.in_ch] + ([a.loop_ch] if a.loop_ch else [])
    kw = {}
    if a.in_dev is not None or a.out_dev is not None:
        kw["device"] = (a.in_dev, a.out_dev)
    pre = int((0.002 if a.loop_ch else 0.010) * fs)
    irs = []
    for r in range(a.repeat):
        print(f"sweep {r + 1}/{a.repeat} ...")
        rec = sd.playrec(play.astype(np.float32).reshape(-1, 1), samplerate=fs,
                         input_mapping=in_map, output_mapping=[a.out_ch],
                         blocking=True, **kw)
        mic = rec[:, 0].astype(float)
        if np.abs(mic).max() > 0.98:
            print("  WARNING: microphone input clipped - lower --level or the mic gain")
        ref = deconvolve(rec[:, 1].astype(float), sweep) if a.loop_ch else None
        irs.append(extract_ir(deconvolve(mic, sweep), ref, int(a.ir_len * fs), pre))
    ir = np.mean(irs, axis=0)
    tail = ir[-int(0.2 * len(ir)):]
    print(f"  peak/noise-floor ~ {20 * np.log10(np.abs(ir).max() / (tail.std() + TINY)):.0f} dB")
    save_meas(a.dir, a.channel, a.pos, ir, fs, bool(a.loop_ch))
    print(f"saved {a.dir}/{a.channel}_{a.pos:02d}.npz")


# ------------------------------------------------------------------ SISO design
def fdw_spectrum(ir, pre, fs, nfft, cycles, tmax, tmin=0.003, nwin=14):
    """Frequency-dependent windowing: window length ~ `cycles` periods at each frequency."""
    f = rfftfreq(nfft, 1 / fs)
    Ts = np.geomspace(tmax, tmin, nwin)
    fk = cycles / Ts
    S = np.empty((nwin, len(f)), complex)
    rise = int(0.001 * fs)
    for k, T in enumerate(Ts):
        L = int(T * fs)
        w = np.zeros(len(ir))
        a0 = max(pre - rise, 0)
        w[a0:pre] = 0.5 - 0.5 * np.cos(np.pi * np.arange(pre - a0) / max(pre - a0, 1))
        flat = int(0.3 * L)
        end = min(pre + L, len(ir))
        w[pre:min(pre + flat, end)] = 1.0
        n = pre + L - (pre + flat)
        fall = 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / max(n, 1))
        s0 = pre + flat
        w[s0:end] = fall[:max(end - s0, 0)]
        S[k] = rfft(ir * w, nfft)
    idx = np.interp(f, fk, np.arange(nwin))
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, nwin - 1)
    fr = idx - lo
    cols = np.arange(len(f))
    return S[lo, cols] * (1 - fr) + S[hi, cols] * fr


def analyse(irs, fs, nfft, a, cal):
    pre = int(0.005 * fs)
    L = pre + int((a.tmax + 0.05) * fs)
    Hs = np.array([fdw_spectrum(crop(ir, int(np.argmax(np.abs(ir))) - pre, L),
                                pre, fs, nfft, a.cycles, a.tmax) for ir in irs])
    f = rfftfreq(nfft, 1 / fs)
    if cal is not None:
        Hs = Hs / (10 ** (curve_at(f, cal) / 20))
    return f, Hs


def prep(f, Hs, a):
    w = smooth_width(f)
    p_raw = np.mean(np.abs(Hs) ** 2, axis=0)
    Hv = Hs.mean(axis=0)
    coh = np.clip(oct_smooth(np.abs(Hv) ** 2 / (p_raw + TINY), 1 / 3), 0, 1)
    M = 10 * np.log10(oct_smooth(p_raw, w) + TINY)
    sel = (f >= a.ref_band[0]) & (f <= a.ref_band[1])
    return dict(Hs=Hs, Hv=Hv, coh=coh, M=M, ref=M[sel].mean(), w=w)


def excess_phase(f, d, fs, nfft, a):
    """Coherence-weighted excess phase of the multi-position vector average (LF only)."""
    coh, wph = d["coh"], 1 / 6
    Hv = d["Hv"] * np.exp(2j * np.pi * f * 0.005)           # undo the 5 ms alignment lead
    mag = np.sqrt(oct_smooth(np.abs(Hv) ** 2, wph))
    mag = np.maximum(mag, 1e-4 * mag.max())
    Es = oct_smooth(Hv * np.exp(-1j * minphase_phase(np.log(mag), nfft)), wph)
    k0 = np.searchsorted(f, a.fmin * 0.7)
    k1 = np.searchsorted(f, a.phase_fmax)
    df = f[1] - f[0]
    gd = -np.angle(Es[k0 + 1:k1] * np.conj(Es[k0:k1 - 1])) / (2 * np.pi * df)
    wts = coh[k0:k1 - 1] * np.abs(Es[k0:k1 - 1]) ** 2 + 1e-30
    tau = wmedian(gd, wts)                                   # residual bulk delay, not corrected
    E2 = Es * np.exp(2j * np.pi * f * tau)
    ph = np.zeros_like(f)
    ph[k0:k1] = np.unwrap(np.angle(E2[k0:k1]))
    ph[k0:k1] -= np.average(ph[k0:k1], weights=coh[k0:k1] ** 2 + 1e-9)
    phw = band_weight(f, a.fmin, a.phase_fmax * 0.5)
    return phw * coh ** 2 * ph


def design_filter(f, d, fs, nfft, a, common_ref):
    """a.phase_mode: 'mixed' (default) | 'min' (minimum phase only) | 'linear' (zero-phase magnitude correction)."""
    tgt = common_ref + target_db(f, a)
    g = band_weight(f, a.fmin, a.fmax)
    M = d["M"]
    so = getattr(a, "smooth_oct", None)
    if so:                                   # user-selected smoothing of the measured response
        M = 10 * np.log10(oct_smooth(np.mean(np.abs(d["Hs"]) ** 2, axis=0), float(so)) + TINY)
    err = (tgt - M) * g
    dl = getattr(a, "dip_limit", 0)
    if dl and dl > 0:                       # DRC-style dip limiting: do not fill deep, narrow nulls
        loc = 10 * np.log10(oct_smooth(10 ** (M / 10), 1 / 3) + TINY)
        allow = np.clip(1 - np.clip(loc - M, 0, None) / dl, 0, 1)
        err = np.where(err > 0, err * allow, err)
    cut = a.max_cut
    if getattr(a, "decay_control", False):  # stronger cuts of low-frequency modal peaks (shorter ringing)
        cut = np.where(f < 200, a.max_cut + 6.0, a.max_cut)
    corr = np.clip(err, -cut, a.max_boost)
    bass_off = 0.0
    if getattr(a, "keep_bass", False):      # keep the low band's average level: cuts and boosts balance out below ~300 Hz
        fsplit = min(300.0, float(a.fmax))
        band = (f >= a.fmin) & (f <= fsplit)
        if band.any():
            taper = np.clip((np.log(fsplit * 1.7) - np.log(np.maximum(f, 1e-3))) / np.log(1.7), 0, 1)
            for _ in range(2):              # re-balance once more after clipping
                bass_off += -float(np.average(corr[band], weights=1.0 / np.maximum(f[band], 1.0)))
                corr = np.clip(err + bass_off * taper * g, -cut, a.max_boost)
            tgt = tgt + bass_off * taper
    corr = oct_smooth(corr, 1 / 12)
    mode = getattr(a, "phase_mode", "mixed")
    logm = corr / 20 * np.log(10)
    if mode == "linear":
        phi = np.zeros_like(f)
        h = np.roll(irfft(np.exp(logm), nfft), a.taps // 2)
    else:
        phi = excess_phase(f, d, fs, nfft, a) if (a.phase_fmax > 0 and mode != "min") else np.zeros_like(f)
        spec = np.exp(logm + 1j * (minphase_phase(logm, nfft) - phi))
        h = np.roll(irfft(spec, nfft), int(a.delay_ms * 1e-3 * fs))
        pm = getattr(a, "pre_ms", None)
        if pm is not None and pm < a.delay_ms:      # DRC-style pre-echo truncation
            p, n = int(np.argmax(np.abs(h))), int(pm * 1e-3 * fs)
            if p > n > 0:
                w = np.ones(len(h))
                w[:p - n] = 0
                w[p - n:p] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
                h = h * w
    loss = 10 * np.log10(np.sum(h[a.taps:] ** 2) / np.sum(h ** 2) + 1e-30)
    h = h[:a.taps].copy()
    n = a.taps // 5
    h[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
    if mode == "linear":
        k = a.taps // 8
        h[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    return h, dict(corr=corr, phi=phi, loss=loss, tgt=tgt, bass_off=bass_off)


def band_report(f, d, Ft, tgt, lo, hi):
    sel = (f >= lo) & (f <= hi)

    def stat(g2):
        db = np.array([10 * np.log10(oct_smooth(np.abs(H) ** 2 * g2, d["w"]) + TINY) for H in d["Hs"]])
        avg = 10 * np.log10(oct_smooth(np.mean(np.abs(d["Hs"]) ** 2, axis=0) * g2, d["w"]) + TINY)
        e_avg = np.sqrt(np.mean((avg - tgt)[sel] ** 2))
        e_pos = np.mean([np.sqrt(np.mean((x - tgt)[sel] ** 2)) for x in db])
        return e_avg, e_pos
    return stat(1.0), stat(np.abs(Ft) ** 2)


def plot_channel(path, f, d, info, Ft, h, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ix = np.unique(np.geomspace(2, len(f) - 1, 1500).astype(int))
    after = 10 * np.log10(oct_smooth(np.mean(np.abs(d["Hs"]) ** 2, axis=0) * np.abs(Ft) ** 2, d["w"]) + TINY)
    fig, ax = plt.subplots(3, 1, figsize=(7, 10))
    for H in d["Hs"]:
        ax[0].semilogx(f[ix], 10 * np.log10(oct_smooth(np.abs(H) ** 2, d["w"]) + TINY)[ix], color="0.8", lw=0.7)
    ax[0].semilogx(f[ix], d["M"][ix], label="before (avg)")
    ax[0].semilogx(f[ix], after[ix], label="predicted after")
    ax[0].semilogx(f[ix], info["tgt"][ix], "k--", lw=0.8, label="target")
    ax[0].set_xlim(20, 20000)
    ax[0].set_ylim(info["tgt"][ix].mean() - 30, info["tgt"][ix].mean() + 15)
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_title("magnitude [dB]")
    ax[1].semilogx(f[ix], 20 * np.log10(np.abs(Ft[ix]) + TINY))
    ax[1].set_xlim(20, 20000)
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].set_title("filter magnitude [dB]")
    t = np.arange(len(h)) / fs * 1000
    ax[2].plot(t, h)
    ax[2].set_title("filter impulse [ms]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_design(a):
    meas = load_meas(a.meas)
    chans = sorted(meas)
    fs = meas[chans[0]][min(meas[chans[0]])][1]
    nfft = nfft_for(a.taps)
    cal = load_curve(a.mic_cal) if a.mic_cal else None
    data = {}
    for ch in chans:
        irs = [meas[ch][p][0] for p in sorted(meas[ch])]
        if len(irs) < 3:
            print(f"WARNING {ch}: only {len(irs)} positions; 5-9 recommended for a stable result")
        f, Hs = analyse(irs, fs, nfft, a, cal)
        data[ch] = prep(f, Hs, a)
    common = np.mean([data[c]["ref"] for c in chans])
    hs, Fts, infos = {}, {}, {}
    for ch in chans:
        hs[ch], infos[ch] = design_filter(f, data[ch], fs, nfft, a, common)
        Fts[ch] = rfft(hs[ch], nfft)
    band = (f >= 20) & (f <= 20000)
    gmax = max(np.abs(Ft[band]).max() for Ft in Fts.values())
    scale = min(1.0, 10 ** (-a.headroom / 20) / gmax)
    os.makedirs(a.out, exist_ok=True)
    for ch in chans:
        d, info = data[ch], infos[ch]
        print(f"\n[{ch}] {len(d['Hs'])} positions, truncation loss {info['loss']:.0f} dB")
        for lo, hi in ((a.fmin, 300), (300, a.fmax)):
            (b_avg, b_pos), (a_avg, a_pos) = band_report(f, d, Fts[ch], info["tgt"], lo, hi)
            print(f"  {lo:.0f}-{hi:.0f} Hz  RMS error vs target:  avg {b_avg:.1f} -> {a_avg:.1f} dB,"
                  f"  per-position {b_pos:.1f} -> {a_pos:.1f} dB")
        h = hs[ch] * scale
        wavfile.write(os.path.join(a.out, f"{ch}.wav"), fs, h.astype(np.float32))
        if a.plot:
            plot_channel(os.path.join(a.out, f"{ch}.png"), f, d, info, Fts[ch], h, fs)
    print(f"\nfilters written to {a.out}/ (fs={fs}, taps={a.taps}, gain {20 * np.log10(scale):.1f} dB, "
          f"latency ~{a.delay_ms:.0f} ms)")


# ------------------------------------------------------------------ MIMO
def cmd_mimo(a):
    meas = load_meas(a.meas)
    chs = a.channels.split(",")
    pos = sorted(set.intersection(*[set(meas[c]) for c in chs]))
    M, N = len(pos), len(chs)
    if M < N:
        sys.exit(f"need at least as many positions ({M}) as channels ({N}); 2x is better")
    fs = meas[chs[0]][pos[0]][1]
    nfft = nfft_for(a.taps)
    f = rfftfreq(nfft, 1 / fs)
    K = np.searchsorted(f, a.fmax * 1.5) + 1
    L = int(a.window * fs)
    win = np.ones(L)
    m = L // 2
    win[-m:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(m) / m)
    H = np.zeros((M, N, K), complex)
    tau = np.zeros((M, N))
    for i, p in enumerate(pos):
        for j, c in enumerate(chs):
            ir = meas[c][p][0]
            tau[i, j] = np.argmax(np.abs(ir)) / fs
            H[i, j] = rfft(crop(ir, 0, L) * win, nfft)[:K]
    fk = f[:K]
    band = (fk >= a.fmin) & (fk <= a.fmax)
    lvl = np.sqrt(np.mean(np.abs(H[:, :, band]) ** 2))
    tgt = 10 ** (target_db(fk, a) / 20) * lvl
    dly = a.delay_ms * 1e-3
    D = tgt[None, :] * np.exp(-2j * np.pi * fk[None, :] * (tau.min(axis=1)[:, None] + dly))
    Hk = np.moveaxis(H, 2, 0)                                   # K x M x N
    A = np.conj(np.swapaxes(Hk, 1, 2)) @ Hk                     # K x N x N
    beta = a.beta * np.mean(np.abs(Hk) ** 2, axis=(1, 2))
    A = A + beta[:, None, None] * np.eye(N)
    rhs = (np.conj(np.swapaxes(Hk, 1, 2)) @ D.T[:, :, None])    # K x N x 1
    W = np.linalg.solve(A, rhs)[:, :, 0]                        # K x N
    W *= band_weight(fk, a.fmin, a.fmax, 0.5, 0.5)[:, None]
    mag = np.abs(W)
    lim = 10 ** (a.max_boost / 20) * np.median(mag[band])
    W *= np.minimum(1.0, lim / (mag + TINY))
    peak = np.abs(W).max()
    W *= 10 ** (-a.headroom / 20) / peak
    os.makedirs(a.out, exist_ok=True)
    Wt = np.zeros((K, N), complex)
    for j, c in enumerate(chs):
        full = np.zeros(nfft // 2 + 1, complex)
        full[:K] = W[:, j]
        h = irfft(full, nfft)[:a.taps].copy()
        n = a.taps // 5
        h[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
        Wt[:, j] = rfft(h, nfft)[:K]
        wavfile.write(os.path.join(a.out, f"{c}.wav"), fs, h.astype(np.float32))

    def stats(Y):
        db = 20 * np.log10(np.abs(Y[:, band]) + TINY)
        err = db - db.mean()                      # flat target, overall level removed
        return np.sqrt(np.mean(err ** 2)), db.std(axis=0).mean()

    print(f"{N} sources, {M} positions, band {a.fmin:.0f}-{a.fmax:.0f} Hz")
    print("                          RMS error vs flat [dB]   spatial spread [dB]")
    for name, Y in ((f"{chs[0]} alone", H[:, 0, :]), ("all, plain sum", H.sum(axis=1)),
                    ("all, optimised", np.einsum('mnk,kn->mk', H, Wt))):
        e, sp = stats(Y)
        print(f"  {name:<22}  {e:8.1f}                {sp:8.1f}")
    print(f"filters written to {a.out}/")


# ------------------------------------------------------------------ apply
def cmd_apply(a):
    fs, x = read_wav(a.inp)
    if x.ndim == 1:
        x = x[:, None]
    fl, hl = read_wav(a.left)
    fr, hr = read_wav(a.right) if a.right else (fl, hl)
    if not (fs == fl == fr):
        sys.exit(f"sample-rate mismatch: audio {fs}, filters {fl}/{fr}")
    y = np.stack([signal.oaconvolve(x[:, c], (hl if c == 0 else hr), mode="full")
                  for c in range(x.shape[1])], axis=1)
    pk = np.abs(y).max()
    if pk > 1:
        print(f"WARNING: output peak {20 * np.log10(pk):.1f} dBFS (float WAV, will not clip but check playback gain)")
    wavfile.write(a.out, fs, y.astype(np.float32))
    print(f"wrote {a.out}")


# ------------------------------------------------------------------ selftest
def synth_ir(fs, seed, delay_ms=6.0, modes=(34, 52, 71, 96, 133), dur=0.9):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    ir = np.zeros(n)
    d0 = int(delay_ms * 1e-3 * fs)
    ir[d0] = 1.0
    for _ in range(30):
        td = rng.uniform(0.004, 0.12)
        ir[d0 + int(td * fs)] += rng.normal() * 0.25 * np.exp(-td / 0.2)
    for fm in modes:
        tau = rng.uniform(0.2, 0.45)
        r = rng.uniform(1.5, 4.0) * rng.choice([-1, 1])
        ir[d0:] += (2 * r / (tau * fs)) * np.exp(-t[:n - d0] / tau) * np.sin(
            2 * np.pi * fm * (1 + 0.01 * rng.normal()) * t[:n - d0])
    sos = np.vstack([signal.butter(2, 45, "hp", fs=fs, output="sos"),
                     signal.butter(2, 18000, "lp", fs=fs, output="sos")])
    return signal.sosfilt(sos, ir)


def simulate_measure(ir, fs, rng, dur=5.0):
    sweep = log_sweep(fs, dur, 15, 20000)
    play = np.concatenate([np.zeros(int(0.3 * fs)), sweep, np.zeros(int(1.2 * fs))])
    lat = 200
    rec = np.concatenate([np.zeros(lat), signal.fftconvolve(play, ir)])[:len(play)]
    rec = rec + 1e-4 * rng.normal(size=len(rec))
    ref = np.concatenate([np.zeros(lat), play])[:len(play)]
    return extract_ir(deconvolve(rec, sweep), deconvolve(ref, sweep), int(0.9 * fs), int(0.002 * fs))


def cmd_selftest(a):
    fs = 48000
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        m, o = os.path.join(tmp, "meas"), os.path.join(tmp, "out")
        err = []
        for ch, base in (("L", 100), ("R", 200)):
            for p in range(1, 6):
                truth = synth_ir(fs, base + p)
                ir = simulate_measure(truth, fs, rng)
                n = min(len(ir), len(truth))
                sel = slice(int(30 * n / fs), int(10000 * n / fs))
                e = 20 * np.log10(np.abs(rfft(ir[:n]))[sel] / np.abs(rfft(truth[:n]))[sel])
                err.append(np.sqrt(np.mean(e ** 2)))
                save_meas(m, ch, p, ir, fs, True)
        print(f"deconvolution accuracy: magnitude RMS error {np.mean(err):.2f} dB (30 Hz-10 kHz)")
        p = build_parser()
        cmd_design(p.parse_args(["design", "--meas", m, "--out", o, "--taps", "16384"]))
        sub_dir = os.path.join(tmp, "sub")
        for c, base, dl in (("subA", 300, 5.0), ("subB", 400, 9.0)):
            for q in range(1, 5):
                truth = synth_ir(fs, base + q, delay_ms=dl)
                save_meas(sub_dir, c, q, simulate_measure(truth, fs, rng, dur=4.0), fs, True)
        print("\n--- MIMO (2 subs, 4 positions) ---")
        cmd_mimo(p.parse_args(["mimo", "--meas", sub_dir, "--channels", "subA,subB",
                               "--out", os.path.join(tmp, "o2"), "--taps", "16384"]))


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    s.add_parser("devices").set_defaults(fn=cmd_devices)

    m = s.add_parser("measure", help="play a log sweep and store one impulse response")
    m.add_argument("--channel", required=True, help="name, e.g. L, R, subA")
    m.add_argument("--pos", type=int, required=True, help="microphone position number")
    m.add_argument("--dir", default="meas")
    m.add_argument("--out-ch", type=int, default=1, help="1-based output channel")
    m.add_argument("--in-ch", type=int, default=1, help="1-based microphone input channel")
    m.add_argument("--loop-ch", type=int, default=0, help="1-based loopback input (needed for MIMO timing)")
    m.add_argument("--in-dev", default=None)
    m.add_argument("--out-dev", default=None)
    m.add_argument("--fs", type=int, default=48000)
    m.add_argument("--dur", type=float, default=8.0)
    m.add_argument("--tail", type=float, default=1.5)
    m.add_argument("--f1", type=float, default=15)
    m.add_argument("--f2", type=float, default=22000)
    m.add_argument("--level", type=float, default=-18, help="dBFS")
    m.add_argument("--repeat", type=int, default=2)
    m.add_argument("--ir-len", type=float, default=1.0)
    m.set_defaults(fn=cmd_measure)

    def common(q):
        q.add_argument("--meas", default="meas")
        q.add_argument("--out", default="filters")
        q.add_argument("--taps", type=int, default=16384)
        q.add_argument("--fmin", type=float, default=25)
        q.add_argument("--max-boost", type=float, default=6)
        q.add_argument("--headroom", type=float, default=1.0)
        q.add_argument("--delay-ms", type=float, default=30)
        q.add_argument("--target-file", default=None, help="text file: freq_Hz dB per line")
        q.add_argument("--bass", type=float, default=0, help="bass lift in dB below --bass-f")
        q.add_argument("--bass-f", type=float, default=100)
        q.add_argument("--tilt", type=float, default=0, help="dB per octave around 1 kHz")

    d = s.add_parser("design", help="compute one FIR per measured channel")
    common(d)
    d.add_argument("--fmax", type=float, default=5000)
    d.add_argument("--phase-fmax", type=float, default=400, help="excess-phase correction limit, 0 = off")
    d.add_argument("--max-cut", type=float, default=15)
    d.add_argument("--cycles", type=float, default=15, help="FDW window length in periods")
    d.add_argument("--tmax", type=float, default=0.5, help="longest FDW window [s]")
    d.add_argument("--ref-band", type=float, nargs=2, default=[300, 3000])
    d.add_argument("--mic-cal", default=None, help="microphone calibration file (freq dB)")
    d.add_argument("--plot", action="store_true")
    d.set_defaults(fn=cmd_design)

    x = s.add_parser("mimo", help="multi-source / multi-mic least-squares filters (e.g. subwoofers)")
    common(x)
    x.set_defaults(delay_ms=40, max_boost=12)
    x.add_argument("--channels", required=True, help="comma list, e.g. subA,subB")
    x.add_argument("--fmax", type=float, default=150)
    x.add_argument("--beta", type=float, default=0.05, help="regularisation (relative)")
    x.add_argument("--window", type=float, default=0.4, help="IR window [s]")
    x.set_defaults(fn=cmd_mimo)

    ap = s.add_parser("apply", help="convolve a WAV with the designed filters (offline audition)")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", default=None)
    ap.set_defaults(fn=cmd_apply)

    s.add_parser("selftest", help="synthetic end-to-end test").set_defaults(fn=cmd_selftest)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
