#!/usr/bin/env python3
"""
Multipoint MIMO room-correction prototype ("support loudspeaker" design).

Idea (same family as Dirac Live ART / Brannmark, Bahne & Ahlen 2013 and
US patent 9,781,510 "variable set of support loudspeakers"):

  * Measure every speaker k at every mic position m:  h[m][k]  (K speakers, M points)
  * For one source channel (e.g. Left) design K FIR filters c_k, one per speaker.
    The primary speaker gets the main filter; the OTHER speakers get "support"
    filters whose only job is to cancel the primary's room decay at all M points.
  * Joint least squares over all points:
        min_c  sum_m || sum_k h[m][k] * c_k  -  d ||^2
               + lam * ||c||^2  + mu * ||c_support||^2
    d = band-limited impulse delayed by a modelling delay.
  * Everything runs in a down-sampled low band (ART works below ~150 Hz).

Run:  python3 tools/mimo_roomcorr.py            (synthetic room demo + metrics)
Use:  design_mimo(h, fs, primary=0, ...) -> list of K FIR filters at low rate
"""
import numpy as np

# ----------------------------------------------------------------------------
# filter design
# ----------------------------------------------------------------------------

def conv_matrix(h, L):
    """Toeplitz matrix T so that T @ c == np.convolve(h, c) for len(c) == L."""
    N = len(h)
    T = np.zeros((N + L - 1, L))
    for i in range(L):
        T[i:i + N, i] = h
    return T


def design_mimo(h, fs, primary, L=256, delay=None, lam=1e-3, mu=1e-2,
                support=None, target=None):
    """
    h        : array (M, K, N) impulse responses, mic m <- speaker k (low-rate)
    fs       : sample rate of h
    primary  : index of the speaker that carries the source channel
    L        : FIR length per speaker
    delay    : modelling delay in samples (default L//2)
    lam      : global regularisation (controls gain / noise amplification)
    mu       : extra penalty on support speakers (0 => support as strong as primary)
    support  : list of speaker indices allowed to act as support (default: all others)
    target   : optional band-limited prototype pulse (centred); default = unit impulse.
               Pass the same low-pass used on h so the LS fit only sees the design band.
    returns  : c (K, L) FIR filters, target d
    """
    M, K, N = h.shape
    if delay is None:
        delay = L // 2
    if support is None:
        support = [k for k in range(K) if k != primary]
    active = [primary] + list(support)

    # target: band-limited (same band as h) unit impulse at the modelling delay
    d = np.zeros(N + L - 1)
    if target is None:
        d[delay] = 1.0
    else:
        half = len(target) // 2
        d[delay - half:delay - half + len(target)] = target

    # stack convolution matrices: rows = all points, cols = all active filters
    A = np.vstack([np.hstack([conv_matrix(h[m, k], L) for k in active])
                   for m in range(M)])
    D = np.tile(d, M)

    # regularisation: lam on everything, extra mu on support speakers
    R = np.full(len(active) * L, lam)
    R[L:] += mu
    normal = A.T @ A + np.diag(R) * np.trace(A.T @ A) / A.shape[1]
    c_act = np.linalg.solve(normal, A.T @ D)

    c = np.zeros((K, L))
    for i, k in enumerate(active):
        c[k] = c_act[i * L:(i + 1) * L]
    return c, d


def apply(h, c):
    """Resulting response at each point: r[m] = sum_k h[m,k] * c[k]."""
    M, K, N = h.shape
    L = c.shape[1]
    r = np.zeros((M, N + L - 1))
    for m in range(M):
        for k in range(K):
            r[m] += np.convolve(h[m, k], c[k])
    return r

# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def decay_db(r, fs, t_ms=100):
    """Energy remaining after t_ms relative to total energy (dB, more negative = better)."""
    out = []
    for x in r:
        i0 = np.argmax(np.abs(x))
        i1 = i0 + int(fs * t_ms / 1000)
        tail = np.sum(x[i1:] ** 2)
        tot = np.sum(x ** 2) + 1e-20
        out.append(10 * np.log10(tail / tot + 1e-12))
    return np.array(out)


def mag_db(r, fs, fmin, fmax, nfft=4096):
    f = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (f >= fmin) & (f <= fmax)
    H = 20 * np.log10(np.abs(np.fft.rfft(r, nfft, axis=-1)) + 1e-12)
    return f[sel], H[:, sel]


def seat_variation_db(r, fs, fmin, fmax):
    """Mean across frequency of the std-dev across points (dB)."""
    _, H = mag_db(r, fs, fmin, fmax)
    return float(np.mean(np.std(H, axis=0)))


def flatness_db(r, fs, fmin, fmax):
    """Mean across points of the std-dev across frequency (dB) = per-seat flatness."""
    _, H = mag_db(r, fs, fmin, fmax)
    return float(np.mean(np.std(H, axis=1)))


def avg_flatness_db(r, fs, fmin, fmax):
    """Std-dev across frequency of the power-averaged response over all points (dB).
    This is what a single (SISO) multipoint filter can flatten; it cannot change
    seat-to-seat differences, only a MIMO design with support speakers can."""
    _, H = mag_db(r, fs, fmin, fmax)
    avg = 10 * np.log10(np.mean(10 ** (H / 10), axis=0))
    return float(np.std(avg))

# ----------------------------------------------------------------------------
# synthetic room (modal model) for the demo
# ----------------------------------------------------------------------------

def modal_room(fs, N, spk_pos, mic_pos, dims=(5.0, 4.0, 2.7), rt60=0.8,
               fmax=150.0, c0=343.0, direct=True):
    """
    Rectangular-room modal sum below fmax (plus a direct-sound path) -> IR.
    Good enough to reproduce 'two speakers share the same room modes'.
    """
    t = np.arange(N) / fs
    ir = np.zeros(N)
    Lx, Ly, Lz = dims
    nmax = int(2 * fmax * max(dims) / c0) + 1
    for nx in range(nmax):
        for ny in range(nmax):
            for nz in range(2):
                f = (c0 / 2) * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2)
                if f == 0 or f > fmax * 1.3:
                    continue
                shape = lambda p: (np.cos(np.pi * nx * p[0] / Lx)
                                   * np.cos(np.pi * ny * p[1] / Ly)
                                   * np.cos(np.pi * nz * p[2] / Lz))
                amp = shape(spk_pos) * shape(mic_pos)
                alpha = 6.9 / rt60
                ir += amp * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t)
    dist = np.linalg.norm(np.array(spk_pos) - np.array(mic_pos))
    if direct:
        n0 = int(round(fs * dist / c0))
        ir[n0] += 1.5 / dist
    return ir


def lowpass(x, fs, fc, taps=129):
    n = np.arange(taps) - taps // 2
    w = np.sinc(2 * fc / fs * n) * np.hamming(taps)
    w /= w.sum()
    return np.convolve(x, w, mode="same")

# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------

def main():
    fs = 1000            # low-rate design (band < 150 Hz, like ART)
    N = 800              # 0.8 s of IR
    fmin, fmax = 25, 150
    speakers = [(1.0, 1.0, 1.0), (4.0, 1.0, 1.0)]          # L, R
    mics = [(2.5, 3.0, 1.2), (2.0, 3.0, 1.2), (3.0, 3.0, 1.2),
            (2.5, 2.5, 1.2), (2.5, 3.5, 1.2), (2.0, 3.5, 1.2), (3.0, 2.5, 1.2),
            (2.5, 3.0, 1.0), (2.5, 3.0, 1.4)]
    M, K = len(mics), len(speakers)

    rng = np.random.default_rng(0)
    h = np.zeros((M, K, N))
    for m, mp in enumerate(mics):
        for k, sp in enumerate(speakers):
            ir = modal_room(fs, N, sp, mp, fmax=fmax)
            ir = lowpass(ir, fs, fmax)
            h[m, k] = ir + rng.normal(0, 1e-3, N)   # a little measurement noise
    h /= np.max(np.abs(h[:, 0]))                 # normalise to the primary's direct sound
    proto = lowpass(np.eye(1, 129, 64)[0], fs, fmax)   # band-limited target pulse

    print(f"synthetic room: {K} speakers, {M} mic points, fs={fs} Hz, band {fmin}-{fmax} Hz\n")

    # 1) uncorrected: Left speaker alone
    c0 = np.zeros((K, 256)); c0[0, 128] = 1.0
    # 2) single-speaker multipoint correction (SISO: only the Left filter)
    c1, _ = design_mimo(h, fs, primary=0, support=[], L=256, target=proto)
    # 3) MIMO: Left as primary, Right as support speaker
    c2, _ = design_mimo(h, fs, primary=0, support=[1], L=256, target=proto)

    rows = []
    for name, c in [("uncorrected (L only)", c0),
                    ("SISO multipoint (L filter only)", c1),
                    ("MIMO multipoint (L + R support)", c2)]:
        r = apply(h, c)
        rows.append((name,
                     avg_flatness_db(r, fs, fmin, fmax),
                     flatness_db(r, fs, fmin, fmax),
                     seat_variation_db(r, fs, fmin, fmax),
                     float(np.mean(decay_db(r, fs, 100))),
                     20 * np.log10(np.max(np.abs(c)) + 1e-12)))

    print(f"{'design':34s} {'avg flat':>9s} {'seat flat':>10s} {'seat var':>9s} {'decay@100ms':>12s} {'peak gain':>10s}")
    for name, af, fl, sv, dc, pk in rows:
        print(f"{name:34s} {af:8.2f}dB {fl:9.2f}dB {sv:8.2f}dB {dc:11.1f}dB {pk:9.1f}dB")
    print("  avg flat : flatness of the power-averaged response (what SISO can fix)")
    print("  seat flat: mean per-seat flatness   seat var: seat-to-seat spread (only MIMO can fix)")

    assert rows[1][1] < rows[0][1], "SISO should flatten the averaged response"
    assert rows[2][2] < rows[1][2], "MIMO should flatten each seat more than SISO"
    assert rows[2][3] < rows[1][3], "MIMO should reduce seat-to-seat variation"
    assert rows[2][4] < rows[1][4], "MIMO should shorten decay more than SISO"
    print("\nOK: support speaker improves flatness, seat variation and decay vs SISO.")


if __name__ == "__main__":
    main()
