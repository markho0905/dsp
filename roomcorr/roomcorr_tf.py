"""roomcorr_tf.py - time/frequency views of an impulse response: spectrogram, waterfall (CSD), decay metrics."""
import numpy as np
from scipy import signal

TINY = 1e-20


def _peak(ir):
    return int(np.argmax(np.abs(ir)))


def spectrogram(ir, fs, fmin=20.0, fmax=20000.0, t_pre=0.02, tmax=0.6, nf=240, nt=240, cycles=6.0):
    """Multi-resolution spectrogram: each frequency uses its own Gaussian window of ~`cycles` periods (2-300 ms,
    varying smoothly), so bass modes are resolved in frequency and treble reflections in time without band edges.
    Returns (t_ms, f, Z_dB rel. the loudest point)."""
    ir = np.asarray(ir, float)
    pk = _peak(ir)
    f = np.geomspace(fmin, min(fmax, fs / 2 * 0.95), nf)
    t = np.linspace(-t_pre, tmax, nt)
    Lmax = int(0.3 * fs)
    a = pk - int(t_pre * fs) - Lmax
    seg = np.zeros(int((t_pre + tmax) * fs) + 2 * Lmax)
    lo, hi = max(a, 0), min(a + len(seg), len(ir))
    seg[lo - a:hi - a] = ir[lo:hi]
    n = np.arange(len(seg))
    centers = Lmax + ((t + t_pre) * fs).astype(int)
    Z = np.empty((nt, nf))
    for j, fc in enumerate(f):
        L = int(np.clip(cycles / fc * fs, 96, Lmax))
        g = signal.windows.gaussian(2 * (L // 2) + 1, L / 6)
        g /= g.sum()
        d = seg * np.exp(-2j * np.pi * fc * n / fs)
        sl = slice(max(centers[0] - L, 0), min(centers[-1] + L, len(seg)))
        y = signal.fftconvolve(d[sl], g, mode="same")
        Z[:, j] = 10 * np.log10(np.abs(y[centers - sl.start]) ** 2 + TINY)
    Z -= Z.max()
    return t * 1000, f, Z


def _smooth_oct(f, P, width):
    """Moving average over `width` octaves (e.g. 1/12)."""
    if width <= 0:
        return P
    lf = np.log2(np.maximum(f, 1e-9))
    out = np.empty_like(P)
    c = np.cumsum(np.concatenate([[0], P]))
    lo = np.searchsorted(lf, lf - 0.5 * width)
    hi = np.searchsorted(lf, lf + 0.5 * width, side="right")
    out[:] = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return out


def waterfall(ir, fs, fmin=15.0, fmax=20000.0, tmax=0.3, nslices=31, win=0.3, rise=0.0005, nf=320, smooth=1 / 12):
    """Cumulative spectral decay: spectra of the response starting at successive times after the direct sound
    (right-side window `win` s).  Returns (t_ms per slice, f, Z_dB rel. first-slice max)."""
    ir = np.asarray(ir, float)
    pk = _peak(ir)
    nr = max(int(rise * fs), 1)
    nw = int(win * fs)
    n = 1 << int(np.ceil(np.log2(nw * 2)))
    fk = np.fft.rfftfreq(n, 1 / fs)
    f = np.geomspace(fmin, min(fmax, fs / 2 * 0.95), nf)
    ts = np.linspace(0, tmax, nslices)
    x = np.concatenate([ir, np.zeros(nw + int(tmax * fs) + nr)])
    tail = np.hanning(2 * max(nw // 4, 1))[max(nw // 4, 1):]
    Z = []
    for t in ts:
        s0 = pk - nr + int(t * fs)
        seg = x[s0:s0 + nw].copy()
        w = np.ones(nw)
        w[:nr] = np.hanning(2 * nr)[:nr]
        w[-len(tail):] = tail
        P = np.abs(np.fft.rfft(seg * w, n)) ** 2
        P = _smooth_oct(fk, P, smooth)
        Z.append(10 * np.log10(np.interp(f, fk, P) + TINY))
    Z = np.array(Z)
    Z -= Z[0].max()
    return ts * 1000, f, Z


def decay_times(t_ms, Z, drop=20.0):
    """Per frequency: first slice time at which the level is `drop` dB below the t=0 level (nan if never)."""
    ref = Z[0]
    out = np.full(Z.shape[1], np.nan)
    for j in range(Z.shape[1]):
        k = np.where(Z[:, j] <= ref[j] - drop)[0]
        if len(k):
            out[j] = t_ms[k[0]]
    return out


# --------------------------------------------------------------------------- room-acoustic metrics of an IR
def schroeder_rt(ir, fs, fc=None, lo=-5.0, hi=-25.0):
    """T20-style reverberation time (s) from the Schroeder integral, noise-truncated; nan if not measurable."""
    x = np.asarray(ir, float)
    if fc:
        sos = signal.butter(4, [fc / 2 ** 0.5, min(fc * 2 ** 0.5, fs / 2 * 0.95)], "bp", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
    p = _peak(x)
    e = x[p:] ** 2
    if len(e) < 4000:
        return float("nan")
    nl = float(np.mean(e[-max(len(e) // 10, 1):]))
    env = np.convolve(e, np.ones(240) / 240, "same")
    below = np.where(env < 2 * nl)[0]
    cut = int(below[0]) if len(below) and below[0] > 1200 else len(e)
    e = np.maximum(e[:cut] - nl, TINY)
    edc = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(edc / edc[0] + TINY)
    i = np.where((db <= lo) & (db >= hi))[0]
    if len(i) < 20:
        return float("nan")
    k = np.polyfit(i / fs, db[i], 1)[0]
    return float(-60.0 / k) if k < 0 else float("nan")


def clarity(ir, fs, ms=50.0):
    """(C50 dB, D50 %) style early/late energy ratio around the direct sound."""
    x = np.asarray(ir, float)
    p = _peak(x)
    e = x[max(p - int(0.001 * fs), 0):p + int(1.0 * fs)] ** 2
    k = int(ms / 1000 * fs)
    early, late = float(e[:k].sum()), float(e[k:].sum()) + TINY
    return 10 * np.log10(early / late), 100 * early / (early + late)


def modal_decay(ir, fs, f1=30.0, f2=250.0, drop=20.0):
    """Median and worst time (ms) for third-octave bands between f1 and f2 to fall `drop` dB (Schroeder)."""
    x = np.asarray(ir, float)
    p = _peak(x)
    out = []
    for fc in np.geomspace(f1, f2, int(round(3 * np.log2(f2 / f1))) + 1):
        sos = signal.butter(3, [fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)], "bp", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, x)[max(p - 120, 0):p + fs]
        edc = np.cumsum((y ** 2)[::-1])[::-1]
        db = 10 * np.log10(edc / (edc[0] + TINY) + TINY)
        k = np.where(db < -drop)[0]
        out.append(k[0] / fs * 1000 if len(k) else 1000.0)
    out = np.array(out)
    return float(np.median(out)), float(out.max())
