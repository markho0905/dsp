"""
roomcorr_dsp.py

* rePhase-style FIR generators: linear/natural-phase crossovers, parametric EQ, all-pass phase EQ, delay
* MSO / Dirac-Bass-Control-style multi-subwoofer optimiser (gain, delay, polarity, 2 all-pass per sub,
  genetic/differential-evolution search over several mic positions) and a least-squares FIR alternative
* bass-management assembly: crossover, sub/main alignment, level match -> multi-output profile
"""
import json
import os

import numpy as np
from scipy import optimize, signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile

import roomcorr as R

TINY = 1e-12


# =============================================================== rePhase-style building blocks
def xo_response(kind, order, fc, f, highpass=False):
    """Complex analog crossover section.  kind: butterworth | linkwitz-riley | bessel."""
    w = 2 * np.pi * np.maximum(np.asarray(f, float), 1e-3)
    wc = 2 * np.pi * fc
    ftype = "highpass" if highpass else "lowpass"
    if kind == "linkwitz-riley":
        if order % 2:
            raise ValueError("Linkwitz-Riley 階數必須是偶數")
        b, a = signal.butter(order // 2, wc, btype=ftype, analog=True)
        h = signal.freqs(b, a, w)[1]
        return h * h
    if kind == "bessel":
        b, a = signal.bessel(order, wc, btype=ftype, analog=True, norm="mag")
    else:
        b, a = signal.butter(order, wc, btype=ftype, analog=True)
    return signal.freqs(b, a, w)[1]


def rbj(kind, f0, q, gain_db, fs):
    """RBJ biquad coefficients (b, a)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    sA = 2 * np.sqrt(A) * al
    if kind == "peaking":
        b = [1 + al * A, -2 * c, 1 - al * A]
        a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind == "lowshelf":
        b = [A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA)]
        a = [(A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA]
    elif kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
        a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    elif kind == "notch":
        b, a = [1, -2 * c, 1], [1 + al, -2 * c, 1 - al]
    elif kind == "allpass":
        b, a = [1 - al, -2 * c, 1 + al], [1 + al, -2 * c, 1 - al]
    else:
        raise ValueError(kind)
    a0 = a[0]
    return np.array(b) / a0, np.array(a) / a0


def allpass_analog(f, f0, q):
    """Closed-form 2nd-order all-pass (used by the optimiser and the final filters alike)."""
    s = 1j * 2 * np.pi * np.asarray(f, float)
    w0 = 2 * np.pi * f0
    return (s * s - (w0 / q) * s + w0 * w0) / (s * s + (w0 / q) * s + w0 * w0)


def minphase(mag, nfft):
    """Minimum-phase complex spectrum with the given magnitude (rfft bins)."""
    lm = np.log(np.maximum(mag, 1e-9))
    return np.exp(lm + 1j * R.minphase_phase(lm, nfft))


def block_response(blk, f, fs, nfft):
    """Complex response of one chain block on rfft bins; linear-phase blocks return a real (zero-phase) response."""
    t = blk["type"]
    ph = blk.get("phase", "natural")
    if t in ("lowpass", "highpass"):
        h = xo_response(blk.get("family", "linkwitz-riley"), int(blk.get("order", 4)), blk["fc"], f, t == "highpass")
        if ph == "linear":
            h = np.abs(h) + 0j
        return -h if blk.get("invert") else h
    if t == "peq":
        b, a = rbj(blk.get("shape", "peaking"), blk["fc"], blk.get("q", 1.0), blk.get("gain", 0.0), fs)
        h = signal.freqz(b, a, worN=2 * np.pi * f / fs)[1]
        return np.abs(h) if ph == "linear" else h
    if t == "allpass":
        return allpass_analog(f, blk["fc"], blk.get("q", 0.7))
    if t == "delay":
        return np.exp(-2j * np.pi * f * blk.get("ms", 0.0) * 1e-3)
    if t == "gain":
        return np.full(len(f), 10 ** (blk.get("db", 0.0) / 20), complex)
    if t == "polarity":
        return np.full(len(f), -1.0, complex)
    raise ValueError(f"unknown block {t}")


def chain_response(chain, fs, nfft):
    """(f, H, has_linear): complex response of the whole chain (without the linear-phase latency)."""
    f = rfftfreq(nfft, 1 / fs)
    H = np.ones(len(f), complex)
    lin = False
    for blk in chain:
        H = H * block_response(blk, f, fs, nfft)
        lin = lin or blk.get("phase") == "linear"
    return f, H, lin


def chain_fir(chain, fs, taps, nfft=None):
    """Render a chain to a FIR.  Linear-phase blocks get a centred latency of taps//2 samples."""
    nfft = nfft or max(1 << 17, 1 << int(np.ceil(np.log2(taps * 4))))
    f, H, lin = chain_response(chain, fs, nfft)
    lat = taps // 2 if lin else 0
    h = irfft(H * np.exp(-2j * np.pi * f * lat / fs), nfft)
    n = taps
    h = h[:n].copy()
    k = max(n // 8, 1)
    h[-k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    if lin:                                   # symmetric fade on the leading edge too
        h[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    return h, lat


def group_delay_at(h, fs, f0, nfft=1 << 17):
    """Group delay (seconds) of FIR h around f0 (median over +-30 %)."""
    H = rfft(h, nfft)
    f = rfftfreq(nfft, 1 / fs)
    sel = (f >= 0.7 * f0) & (f <= 1.3 * f0)
    d = np.angle(H[1:] * np.conj(H[:-1]))
    gd = -d / (2 * np.pi * (f[1] - f[0]))
    return float(np.median(gd[:-1][sel[1:-1]]))


# =============================================================== multi-sub optimiser
class BassData:
    """Complex sub responses H[p, n, k] on selected FFT bins, plus helper grids."""

    def __init__(self, irs, fs, fmin, fmax, nfft=1 << 17, window=0.35, npts=260, dense=False):
        # irs: {sub_name: {pos: ir}}
        self.names = sorted(irs)
        self.pos = sorted(set.intersection(*[set(irs[n]) for n in self.names]))
        self.fs, self.nfft = fs, nfft
        f_all = rfftfreq(nfft, 1 / fs)
        if dense:
            idx = np.arange(int(max(fmin, f_all[1]) / f_all[1]), int(fmax / f_all[1]) + 1)
        else:
            idx = np.unique((np.geomspace(max(fmin, f_all[1]), fmax, npts) / f_all[1]).astype(int))
        self.idx = idx[(idx > 0) & (idx < len(f_all))]
        self.f = f_all[self.idx]
        L = int(window * fs)
        win = np.ones(L)
        m = L // 2
        win[-m:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(m) / m)
        self.H = np.zeros((len(self.pos), len(self.names), len(self.f)), complex)
        for i, p in enumerate(self.pos):
            for j, n in enumerate(self.names):
                self.H[i, j] = rfft(R.crop(irs[n][p], 0, L) * win, nfft)[self.idx]
        self.irs = irs
        self.win, self.L = win, L

    def full_spec(self, ir):
        """Complex spectrum of one IR on the whole rfft grid (same window as the optimiser data)."""
        return rfft(R.crop(ir, 0, self.L) * self.win, self.nfft)


class BassOptimizer:
    """MSO / Dirac-Bass-Control-style optimiser: per-sub gain, delay, polarity, two all-pass filters."""

    def __init__(self, data, fmin, fmax, target_db=None, max_delay_ms=20.0, n_allpass=2,
                 gain_range=(-12.0, 3.0), mlp_weight=0.5, level_weight=0.3):
        self.d = data
        self.band = (data.f >= fmin) & (data.f <= fmax)
        self.f = data.f[self.band]
        self.H = data.H[:, :, self.band]
        self.S = self.H.shape[1]
        self.tgt = np.zeros_like(self.f) if target_db is None else np.interp(self.f, target_db[0], target_db[1])
        self.max_delay, self.nap = max_delay_ms, n_allpass
        self.gr, self.mw, self.lw = gain_range, mlp_weight, level_weight
        self.fmax = fmax
        self.ref_level = 20 * np.log10(np.abs(self.H).mean(axis=0).sum(axis=0).mean() + TINY)
        self.stride = 3 + 2 * n_allpass

    # parameter vector per sub: [gain dB, delay ms, polarity(0/1), (f_ap, q_ap) * n_allpass]
    def bounds(self):
        b = []
        for n in range(self.S):
            b += [self.gr, (0.0, 1e-6) if n == 0 else (0.0, self.max_delay), (0, 1)]
            for _ in range(self.nap):
                b += [(self.f[0] * 0.8, self.fmax * 1.3), (0.3, 1.0)]
        return b

    def sub_response(self, x, n, f=None):
        f = self.f if f is None else f
        o = n * self.stride
        g = 10 ** (x[o] / 20) * (-1 if x[o + 2] > 0.5 else 1)
        G = g * np.exp(-2j * np.pi * f * x[o + 1] * 1e-3)
        for k in range(self.nap):
            G = G * allpass_analog(f, x[o + 3 + 2 * k], x[o + 4 + 2 * k])
        return G

    def dense_G(self, x, f, eq=None):
        """Per-sub complex filter responses on an arbitrary grid f: list of (name, G)."""
        out = []
        for n in range(self.S):
            G = self.sub_response(x, n, f)
            if eq is not None:
                G = G * eq
            out.append((self.d.names[n], G))
        return out

    def combined(self, x):
        Y = 0
        for n in range(self.S):
            Y = Y + self.H[:, n, :] * self.sub_response(x, n)[None, :]
        return Y

    def cost(self, x):
        Y = self.combined(x)
        db = 20 * np.log10(np.abs(Y) + TINY)
        dev = db - self.tgt[None, :]
        dev = dev - dev.mean()
        c = np.sqrt(np.mean(dev ** 2)) + self.mw * np.sqrt(np.mean(dev[0] ** 2))
        lvl = 10 * np.log10(np.mean(np.abs(Y) ** 2) + TINY)
        return c + self.lw * max(0.0, self.ref_level - lvl)

    def run(self, iters=250, pop=18, seed=1, progress=None):
        integ = [(i % self.stride) == 2 for i in range(self.S * self.stride)]
        cb = None
        if progress:
            cb = lambda xk, convergence=0: progress(f"最佳化中…（收斂 {convergence * 100:.0f}%）")  # noqa: E731
        res = optimize.differential_evolution(self.cost, self.bounds(), maxiter=iters, popsize=pop, seed=seed,
                                              tol=1e-7, mutation=(0.4, 1.0), recombination=0.8, polish=False,
                                              integrality=integ, callback=cb, updating="immediate")
        x = res.x.copy()
        # local polish on the continuous parameters (polarity fixed)
        free = [i for i in range(len(x)) if not integ[i] and self.bounds()[i][1] > 1e-3]
        bnds = self.bounds()

        def f2(z):
            y = x.copy()
            y[free] = z
            return self.cost(y)
        r2 = optimize.minimize(f2, x[free], method="Powell", bounds=[bnds[i] for i in free],
                               options=dict(maxiter=2000, xtol=1e-3, ftol=1e-6))
        if r2.fun < res.fun:
            x[free] = r2.x
        self.x = x
        return x

    # ---- results
    def params_table(self, x=None):
        x = self.x if x is None else x
        rows = []
        for n in range(self.S):
            o = n * self.stride
            ap = [(round(float(x[o + 3 + 2 * k]), 1), round(float(x[o + 4 + 2 * k]), 2)) for k in range(self.nap)]
            rows.append(dict(name=self.d.names[n], gain_db=float(x[o]), delay_ms=float(x[o + 1]),
                             invert=bool(x[o + 2] > 0.5), allpass=ap))
        return rows

    def stats(self, x=None):
        Y = self.combined(self.x if x is None else x)
        db = 20 * np.log10(np.abs(Y) + TINY)
        dev = db - self.tgt[None, :]
        dev = dev - dev.mean()
        return dict(rms=float(np.sqrt(np.mean(dev ** 2))), spread=float(db.std(axis=0).mean()),
                    mlp=float(np.sqrt(np.mean((dev[0] - dev[0].mean()) ** 2))))

    def plain_x(self):
        x = np.zeros(self.S * self.stride)
        for n in range(self.S):
            o = n * self.stride
            for k in range(self.nap):
                x[o + 3 + 2 * k] = self.fmax * 1.3        # all-pass far above the band = inert
                x[o + 4 + 2 * k] = 1.0
        return x


def bass_eq(Hsum_p, f, tgt_db, fmin, fmax, max_boost=3.0, max_cut=12.0, width=1 / 6):
    """Common minimum-phase EQ (dB on grid f) flattening the position-averaged response of the optimised subs."""
    p = np.mean(np.abs(Hsum_p) ** 2, axis=0)
    m = 10 * np.log10(R.oct_smooth(p, width) + TINY)
    err = tgt_db - m
    err -= np.median(err[(f >= fmin) & (f <= fmax)])
    g = R.band_weight(f, fmin, fmax, 0.5, 0.5)
    return np.clip(err * g, -max_cut, max_boost)


def bass_ls(data, fmin, fmax, tgt_db=None, delay_ms=40.0, beta=0.05, max_boost=12.0):
    """Least-squares FIR matrix solution W[k, n] on data.f (Bass-Control-like, complex per-frequency)."""
    H = data.H
    f = data.f
    M, N, K = H.shape
    tau = np.array([[np.argmax(np.abs(data.irs[nm][p])) / data.fs for nm in data.names] for p in data.pos])
    lvl = np.sqrt(np.mean(np.abs(H[:, :, (f >= fmin) & (f <= fmax)]) ** 2))
    tg = 10 ** ((np.zeros(K) if tgt_db is None else np.interp(f, tgt_db[0], tgt_db[1])) / 20) * lvl
    D = tg[None, :] * np.exp(-2j * np.pi * f[None, :] * (tau.min(axis=1)[:, None] + delay_ms * 1e-3))
    Hk = np.moveaxis(H, 2, 0)
    A = np.conj(np.swapaxes(Hk, 1, 2)) @ Hk
    A = A + (beta * np.mean(np.abs(Hk) ** 2, axis=(1, 2)))[:, None, None] * np.eye(N)
    W = np.linalg.solve(A, np.conj(np.swapaxes(Hk, 1, 2)) @ D.T[:, :, None])[:, :, 0]
    band = (f >= fmin) & (f <= fmax)
    mag = np.abs(W)
    lim = 10 ** (max_boost / 20) * np.median(mag[band])
    return W * np.minimum(1.0, lim / (mag + TINY))[:, :] * R.band_weight(f, fmin, fmax, 0.5, 0.5)[:, None]


def crossover_align(Ysub, Ymain, f, fc, xo_kind="linkwitz-riley", order=4, linear=True, span_ms=12.0):
    """Search sub delay (ms, negative = delay the mains instead) and polarity maximising the summed response
    around the crossover.  Ysub / Ymain: complex responses (before the crossover) on grid f."""
    lp = xo_response(xo_kind, order, fc, f, False)
    hp = xo_response(xo_kind, order, fc, f, True)
    if linear:
        lp, hp = np.abs(lp), np.abs(hp)
    sel = (f >= 0.6 * fc) & (f <= 1.7 * fc)
    best = None
    for pol in (1, -1):
        for d in np.arange(-span_ms, span_ms + 1e-9, 0.05):
            s = pol * Ysub * lp * np.exp(-2j * np.pi * f * d * 1e-3) + Ymain * hp
            ref = np.abs(Ysub * lp) + np.abs(Ymain * hp) + TINY
            score = np.mean(np.abs(s[sel]) / ref[sel])           # 1.0 = perfectly in phase
            if best is None or score > best[0]:
                best = (score, d, pol)
    return dict(score=float(best[0]), delay_ms=float(best[1]), polarity=int(best[2]))


# =============================================================== multi-output profile
def save_multi_profile(folder, fs, outputs):
    """outputs: list of dict(name, channel(1-based), paths=[(input_index, h, gain)]).  Writes wavs + profile.json."""
    os.makedirs(folder, exist_ok=True)
    meta = {"version": 2, "fs": int(fs), "outputs": []}
    for o in outputs:
        fn = f"{o['name']}.wav"
        wavfile.write(os.path.join(folder, fn), fs, np.asarray(o["h"], np.float32))
        meta["outputs"].append(dict(name=o["name"], channel=int(o["channel"]),
                                    paths=[dict(input=int(i), file=fn, gain=float(g)) for i, g in o["paths"]]))
    json.dump(meta, open(os.path.join(folder, "profile.json"), "w"), indent=1, ensure_ascii=False)


def build_bass_system(fs, taps, fc, xo_kind, xo_order, xo_phase, sub_G, main_h, align, sub_trim_db,
                      out_channels, nfft=None):
    """Assemble the output set.
    sub_G: list of (name, G) with G the complex sub response filter on rfftfreq(nfft) (gain, delay, all-pass, EQ ...).
    main_h: {'L': h, 'R': h} room-correction FIRs (or None -> impulse); align: dict(delay_ms, polarity).
    Returns (outputs, info)."""
    nfft = nfft or max(1 << 17, 1 << int(np.ceil(np.log2(taps * 4))))
    f = rfftfreq(nfft, 1 / fs)
    lin = xo_phase == "linear"
    lat_xo = int(min(3.0 / fc * fs, taps // 2 - 1)) if lin else 0
    lp = xo_response(xo_kind, xo_order, fc, f, False)
    hp = xo_response(xo_kind, xo_order, fc, f, True)
    if lin:
        lp, hp = np.abs(lp), np.abs(hp)
    lat_ph = np.exp(-2j * np.pi * f * lat_xo / fs)
    imp = np.zeros(taps)
    imp[0] = 1.0
    main_h = main_h or {"L": imp, "R": imp}
    d_ms = align["delay_ms"]
    sub_delay, main_delay = max(d_ms, 0.0), max(-d_ms, 0.0)
    gd = float(np.mean([group_delay_at(main_h[c], fs, fc) for c in "LR"]))
    pad = max(gd, 0.0)
    outs = []
    for c, ch in (("L", out_channels[0]), ("R", out_channels[1])):
        Hm = rfft(main_h[c], nfft) * hp * lat_ph * np.exp(-2j * np.pi * f * main_delay * 1e-3)
        outs.append(dict(name=c, channel=ch, h=_fade(irfft(Hm, nfft)[:taps]), paths=[(0 if c == "L" else 1, 1.0)]))
    trim = 10 ** (sub_trim_db / 20) * align["polarity"]
    for n, (name, G) in enumerate(sub_G):
        Hs = G * trim * lp * lat_ph * np.exp(-2j * np.pi * f * (sub_delay * 1e-3 + pad))
        outs.append(dict(name=name, channel=out_channels[2 + n], h=_fade(irfft(Hs, nfft)[:taps]),
                         paths=[(0, 0.5), (1, 0.5)]))
    return outs, dict(lat_xo_ms=lat_xo / fs * 1000, main_gd_ms=gd * 1000, sub_pad_ms=pad * 1000)


def _fade(h):
    h = h.copy()
    k = max(len(h) // 8, 1)
    h[-k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    return h
