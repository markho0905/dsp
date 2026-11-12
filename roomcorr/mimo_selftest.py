#!/usr/bin/env python3
"""mimo_selftest.py - synthetic two-speaker room: compare the SISO design with the MIMO joint design.

No audio hardware needed.  Builds a rectangular-room modal model (both speakers share the same room modes, which
is what makes cross-speaker support useful), "measures" L and R at the three standard positions, runs
roomcorr_analysis.Analysis exactly as the app does, and prints per-speaker / combined errors, position-to-position
spread and low-frequency decay for: uncorrected, SISO (MIMO off) and MIMO on.

    python3 mimo_selftest.py            # default room
    python3 mimo_selftest.py --npos 5   # also measure mains at the extra positions 4..npos
"""
import argparse
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roomcorr as R                      # noqa: E402
import roomcorr_analysis as A             # noqa: E402
import roomcorr_tf as T                   # noqa: E402

C0 = 343.0


def room_ir(fs, spk, mic, dims=(5.2, 4.1, 2.7), rt60_lf=0.75, rt60_hf=0.35, fmodal=400.0, n=None, seed=0,
            lf_lift_db=4.0):
    """Modal sum below `fmodal` (shared by every speaker/mic pair) + direct sound, image-source early reflections
    and a diffuse tail above it (Linkwitz-Riley split).  Absolute timing (propagation delay included)."""
    n = n or int(1.2 * fs)
    t = np.arange(n) / fs
    Lx, Ly, Lz = dims
    d0 = np.linalg.norm(np.array(spk) - np.array(mic)) / C0
    kd = int(round(d0 * fs))
    # ---- modal part
    modal = np.zeros(n)
    nmax = int(2 * fmodal * max(dims) / C0) + 1
    alpha = 6.9 / rt60_lf
    tt = np.clip(t - d0, 0, None)
    on = (t >= d0)
    for nx in range(nmax):
        for ny in range(nmax):
            for nz in range(3):
                f = (C0 / 2) * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2)
                if f == 0 or f > fmodal:
                    continue
                eps = (2 if nx else 1) * (2 if ny else 1) * (2 if nz else 1)
                shape = lambda p: (np.cos(np.pi * nx * p[0] / Lx) * np.cos(np.pi * ny * p[1] / Ly)
                                   * np.cos(np.pi * nz * p[2] / Lz))
                modal += eps * shape(spk) * shape(mic) * np.exp(-alpha * tt) * np.cos(2 * np.pi * f * tt) * on
    lp = signal.butter(2, fmodal, "lp", fs=fs, output="sos")
    modal = signal.sosfilt(lp, signal.sosfilt(lp, modal))                     # LR4 low-pass, causal
    # ---- direct + image sources (orders 0-2) + diffuse tail, above fmodal
    rng = np.random.default_rng(seed)
    hf = np.zeros(n)
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                order = abs(ix) + abs(iy) + abs(iz)
                if order > 2:
                    continue
                sx = (ix * Lx + (spk[0] if ix % 2 == 0 else Lx - spk[0])) if ix else spk[0]
                sy = (iy * Ly + (spk[1] if iy % 2 == 0 else Ly - spk[1])) if iy else spk[1]
                sz = (iz * Lz + (spk[2] if iz % 2 == 0 else Lz - spk[2])) if iz else spk[2]
                d = np.linalg.norm(np.array([sx, sy, sz]) - np.array(mic))
                k = int(round(d / C0 * fs))
                if k < n:
                    hf[k] += (0.8 ** order) / d * (1.0 if order == 0 else rng.choice([-1, 1]))
    refl = hf.copy()
    refl[kd] = 0.0
    refl = signal.sosfilt(signal.butter(1, 4000, "lp", fs=fs, output="sos"), refl)   # reflections lose HF
    hf = (hf - (hf - refl)) * 0 + refl
    hf[kd] += 1.0 / (d0 * C0)
    tail = rng.normal(size=n) * np.exp(-6.9 / rt60_hf * t) * 0.03 / (d0 * C0)
    tail[:kd + int(0.012 * fs)] = 0
    hf = hf + tail
    hp = signal.butter(2, fmodal, "hp", fs=fs, output="sos")
    hf = signal.sosfilt(hp, signal.sosfilt(hp, hf))                           # LR4 high-pass, causal
    # ---- level calibration: modal band sits `lf_lift_db` above the mid band of the direct/reflection part
    nf = 1 << 16
    fk = np.fft.rfftfreq(nf, 1 / fs)
    Pm = np.abs(np.fft.rfft(modal[:nf], nf)) ** 2
    Ph = np.abs(np.fft.rfft(hf[:nf], nf)) ** 2
    lf = Pm[(fk >= 25) & (fk <= 300)].mean()
    mf = Ph[(fk >= 300) & (fk <= 3000)].mean()
    modal *= np.sqrt(mf / lf) * 10 ** (lf_lift_db / 20)
    ir = modal + hf
    return ir / (np.abs(ir).max() * 2.0)


def build_session(fs, npos=3, dims=(5.2, 4.1, 2.7)):
    spk = {"L": (1.3, 1.0, 1.0), "R": (3.9, 1.0, 1.0)}
    mics = {1: (2.6, 3.0, 1.15), 2: (2.3, 3.0, 1.15), 3: (2.9, 3.0, 1.15),
            4: (2.6, 2.6, 1.15), 5: (2.6, 3.4, 1.15), 6: (2.2, 3.4, 1.1), 7: (3.0, 2.6, 1.2)}
    meas = {"L": {}, "R": {}}
    for pos in range(1, npos + 1):
        full = {c: room_ir(fs, spk[c], mics[pos], dims, seed=10 * pos + i) for i, c in enumerate("LR")}
        anchor = min(int(np.argmax(np.abs(full[c]))) for c in "LR") - int(0.010 * fs)   # as Session.save does
        for c in "LR":
            meas[c][pos] = R.crop(full[c], anchor, int(1.0 * fs))
    return meas


def achieved_irs(an, ds, pos):
    """Predicted IRs at a position for source L, source R and L+R, including cross paths if present."""
    hx = getattr(ds, "hx", None) or {}
    irL, irR = an.meas["L"][pos], an.meas["R"][pos]
    out = {}
    for src in "LR":
        own = signal.fftconvolve(irL if src == "L" else irR, ds.h[src])
        x = hx.get(src + ("R" if src == "L" else "L"))
        if x is not None:
            other = signal.fftconvolve(irR if src == "L" else irL, x)
            n = max(len(own), len(other))
            own = np.pad(own, (0, n - len(own))) + np.pad(other, (0, n - len(other)))
        out[src] = own
    n = max(len(out["L"]), len(out["R"]))
    out["LR"] = np.pad(out["L"], (0, n - len(out["L"]))) + np.pad(out["R"], (0, n - len(out["R"])))
    return out


def metrics(an, ds_or_none, fmin=25.0, fmax=300.0, nfft=1 << 16):
    """Per-source RMS error vs flat (dB, level removed), spread across positions, decay (-20 dB time) 30-250 Hz."""
    fs = an.fs
    f = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (f >= fmin) & (f <= fmax)
    res = {}
    for src in ("L", "R", "LR"):
        dbs, decs = [], []
        for pos in sorted(an.pairs):
            if ds_or_none is None:
                irL, irR = an.meas["L"][pos], an.meas["R"][pos]
                ir = {"L": irL, "R": irR, "LR": irL + irR}[src]
            else:
                ir = achieved_irs(an, ds_or_none, pos)[src]
            p = int(np.argmax(np.abs(ir)))
            seg = ir[max(p - int(0.005 * fs), 0):p + int(0.5 * fs)]
            P = np.abs(np.fft.rfft(seg, nfft)) ** 2
            P = R.oct_smooth(P, 1 / 6)
            dbs.append(10 * np.log10(P + 1e-20))
            med, mx = T.modal_decay(ir, fs, 30.0, 250.0, 20.0)
            decs.append(med)
        dbs = np.array(dbs)[:, sel]
        err = dbs - dbs.mean(axis=1, keepdims=True)
        res[src] = dict(rms=float(np.sqrt(np.mean(err ** 2))), spread=float(dbs.std(axis=0).mean()),
                        decay=float(np.mean(decs)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npos", type=int, default=3)
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--band", type=float, nargs=2, default=[25.0, 300.0])
    a = ap.parse_args()
    fs = a.fs
    meas = build_session(fs, a.npos)
    an = A.Analysis(meas, {}, fs)
    base = A.slot_settings("std")
    rows = [("uncorrected", None)]
    for name, extra in (("SISO (MIMO off)", dict(mimo=False)), ("MIMO 空間一致", dict(mimo=True, mimo_mode="space")),
                        ("MIMO 餘音控制", dict(mimo=True, mimo_mode="decay"))):
        s = dict(base, **extra)
        ds = an.design(A.params_from_settings(s))
        rows.append((name, ds))
    lo, hi = a.band
    print(f"synthetic room, {len(an.pairs)} positions, band {lo:.0f}-{hi:.0f} Hz "
          f"(RMS vs flat / spread across positions / -20 dB decay 30-250 Hz)\n")
    print(f"{'design':18s} | {'L alone':>24s} | {'R alone':>24s} | {'L+R':>24s}")
    for name, ds in rows:
        m = metrics(an, ds, lo, hi)
        cells = [f"{m[k]['rms']:4.1f} dB {m[k]['spread']:4.1f} dB {m[k]['decay']:5.0f} ms" for k in ("L", "R", "LR")]
        gain = f"  filter gain {20 * np.log10(ds.scale):5.1f} dB" if ds is not None else ""
        print(f"{name:18s} | " + " | ".join(cells) + gain)
        if ds is not None:
            for line in an.design_text(ds):
                print(" " * 21 + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
