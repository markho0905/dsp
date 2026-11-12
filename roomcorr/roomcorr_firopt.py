"""roomcorr_firopt.py - prediction model and one-click optimiser for the FIR tool.

The FIR tool adds a chain of blocks (PEQ, all-pass, delay, gain ...) on top of the current correction.  This module
predicts what the listener gets (measurement x current filter x chain) and searches a chain that brings it closest to
the target while also weighing modal decay (standing waves / RT), position-to-position reliability, the speech band,
left/right phase coherence, group delay and channel balance.

General method (no room-specific frequencies):
* low frequencies (< f_iir): natural-phase (minimum-phase) IIR peaking filters - for minimum-phase modes they shorten
  the decay as well as flatten the level; high frequencies: linear-phase FIR peaking filters, which change level
  without adding phase distortion.
* error weighting: peaks count fully, dips only partly (nulls cannot be filled), bands where the positions disagree
  count less, long-decaying bands count more, 300 Hz - 4 kHz (speech) counts more.
* L/R: cut-only gain to match the mid-band level, then delay + up to two all-pass sections on one channel to maximise
  the coherent L+R sum at the listening position, penalising added group delay.
"""
import numpy as np
from scipy import optimize, signal
from scipy.fft import rfft

import roomcorr_dsp as D
import roomcorr_tf as T

TINY = 1e-12


def chain_for(chain, ch):
    """Blocks that apply to channel ch ("L"/"R"); blocks without "ch" apply to both."""
    return [b for b in chain if b.get("ch", "LR") in ("LR", ch)]


def _box_smooth(f, x, width):
    lf = np.log2(np.maximum(f, 1e-9))
    c = np.cumsum(np.concatenate([[0], x]))
    lo = np.searchsorted(lf, lf - width / 2)
    hi = np.searchsorted(lf, lf + width / 2, side="right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _biquad_db(kind, fc, q, g, fs, f):
    b, a = D.rbj(kind, fc, q, g, fs)
    z = np.exp(-2j * np.pi * f / fs)
    h = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return 20 * np.log10(np.abs(h) + TINY)


class Model:
    """Everything the FIR tool needs to draw and optimise quickly, on a log-frequency grid."""

    def __init__(self, an, hbase, tgt_rel, fs, fmax=20000.0, n=480, hx=None):
        self.an, self.fs = an, fs
        self.hbase = {c: np.asarray(hbase[c], float) for c in "LR"}
        self.hx = {k: np.asarray(v, float) for k, v in (hx or {}).items()}     # MIMO cross filters: "LR", "RL"
        f = an.f
        self.fg = fg = np.geomspace(15.0, min(20000.0, an.fs / 2 * 0.95), n)
        self.fmax = float(fmax)
        self.tgt = np.interp(fg, f, tgt_rel)
        F = {c: rfft(self.hbase[c], an.nfft) for c in "LR"}
        Fx = {k: rfft(v, an.nfft) for k, v in self.hx.items()}
        self.base, self.spread, self.center = {}, {}, {}
        lat = {c: int(np.argmax(np.abs(self.hbase[c]))) / an.fs for c in "LR"}
        offs = []
        for c in "LR":
            oth = "R" if c == "L" else "L"
            if (c + oth) in Fx and an.pairs:              # source c reaches the listener through both speakers
                prs = [an.pairs[k] for k in sorted(an.pairs)]
                P = np.abs(np.stack([q["H" + c] * F[c] + q["H" + oth] * Fx[c + oth] for q in prs])) ** 2
            else:
                Hs = np.asarray(an.data[c]["Hs"])
                P = np.abs(Hs * F[c][None, :]) ** 2
            pm = _box_smooth(f, P.mean(axis=0), 1 / 12)
            self.base[c] = 10 * np.log10(np.interp(fg, f, pm) + TINY)
            dbs = np.array([10 * np.log10(np.interp(fg, f, _box_smooth(f, p, 1 / 6)) + TINY) for p in P])
            self.spread[c] = dbs.std(axis=0) if len(dbs) > 1 else np.zeros(n)
            mid = (fg >= 200) & (fg <= 2000)
            offs.append(np.median(self.base[c][mid] - self.tgt[mid]))
        off = float(np.mean(offs))                      # one common offset keeps the real L/R level difference
        for c in "LR":
            self.base[c] = self.base[c] - off
        self.has_center = an.center is not None
        if self.has_center:
            cpos = an.center
            for c in "LR":
                oth = "R" if c == "L" else "L"
                Hc = cpos["H" + c] * F[c] + (cpos["H" + oth] * Fx[c + oth] if (c + oth) in Fx else 0.0)
                tau = cpos["tau"] + lat[c]
                Hc = Hc * np.exp(2j * np.pi * f * tau)  # remove the bulk delay (keeps the L/R difference below)
                re = np.interp(fg, f, _box_smooth(f, Hc.real, 1 / 48))
                im = np.interp(fg, f, _box_smooth(f, Hc.imag, 1 / 48))
                self.center[c] = re + 1j * im
            # put back the real arrival difference between the channels
            dT = (cpos["dT"] / an.fs) + (lat["R"] - lat["L"])
            self.center["R"] = self.center["R"] * np.exp(-2j * np.pi * fg * dT)
            self.center_ir = {c: an.meas[c][cpos["pos"]] for c in "LR"}
            self.center_ir_x = {}
            for c in "LR":                           # source c -> other speaker (MIMO support) at the centre position
                oth = "R" if c == "L" else "L"
                if (c + oth) in self.hx:
                    self.center_ir_x[c] = signal.fftconvolve(an.meas[oth][cpos["pos"]], self.hx[c + oth])
        # modal decay weighting from the predicted (current filter) response at the listening position
        self.decay_w = np.ones(n)
        if self.has_center:
            dw = []
            for c in "LR":
                ir = self.source_ir(c)
                t, fw, Z = T.waterfall(ir, an.fs, 15.0, 600.0, tmax=0.4, nslices=21, win=0.3, smooth=1 / 12, nf=200)
                d = T.decay_times(t, Z, 20.0)
                d = np.where(np.isnan(d), t[-1], d)
                dw.append(np.interp(fg, fw, d, right=d[-1]))
            d = np.mean(dw, axis=0)
            lowb = (fg >= 20) & (fg <= 500)
            med = float(np.median(d[lowb])) or 1.0
            self.decay_w = np.where(fg < 500, np.clip(d / med, 1.0, 3.0), 1.0)
            self.decay_ms = d

    def source_ir(self, c):
        """Predicted IR at the centre for source c through the current filters (own speaker + MIMO support)."""
        ir = signal.fftconvolve(self.center_ir[c], self.hbase[c])
        x = getattr(self, "center_ir_x", {}).get(c)
        if x is not None:
            n = max(len(ir), len(x))
            ir = np.pad(ir, (0, n - len(ir))) + np.pad(x, (0, n - len(x)))
        return ir

    # ---- responses
    def chain_H(self, chain, f=None):
        f = self.fg if f is None else f
        H = np.ones(len(f), complex)
        for b in chain:
            H = H * D.block_response(b, f, self.fs, 0)
        return H

    def mag(self, ch, chain):
        return self.base[ch] + 20 * np.log10(np.abs(self.chain_H(chain_for(chain, ch))) + TINY)

    def center_resp(self, ch, chain):
        if not self.has_center:
            return None
        return self.center[ch] * self.chain_H(chain_for(chain, ch))

    def sum_resp(self, chain):
        if not self.has_center:
            return None
        return self.center_resp("L", chain) + self.center_resp("R", chain)

    def gd_ms(self, H):
        ph = np.unwrap(np.angle(H))
        gd = -np.gradient(ph, 2 * np.pi * self.fg) * 1000
        return _box_smooth(self.fg, gd, 1 / 6)

    # ---- metrics (used for the report)
    def metrics(self, chain, taps):
        fg = self.fg
        out = {}

        def rms(ch, lo, hi):
            s = (fg >= lo) & (fg <= min(hi, self.fmax))
            e = self.mag(ch, chain)[s] - self.tgt[s]
            return float(np.sqrt(np.mean((e - np.mean(e)) ** 2)))
        for c in "LR":
            out[f"rms_lo_{c}"], out[f"rms_mid_{c}"], out[f"rms_hi_{c}"] = rms(c, 20, 200), rms(c, 200, 2000), rms(c, 2000, 20000)
        dl = self.mag("L", chain) - self.mag("R", chain)
        for name, lo, hi in (("lr_lo", 20, 200), ("lr_mid", 200, 2000), ("lr_hi", 2000, 20000)):
            s = (fg >= lo) & (fg <= hi)
            out[name] = float(np.mean(np.abs(dl[s])))
        mid = (fg >= 500) & (fg <= 2000)
        out["bal"] = float(np.mean(dl[mid]))
        if self.has_center:
            a, b = self.center_resp("L", chain), self.center_resp("R", chain)
            s = (fg >= 40) & (fg <= 400)
            out["coh"] = float(np.sum(np.abs(a + b)[s] ** 2) / np.sum((np.abs(a) + np.abs(b))[s] ** 2) * 100)
            s2 = (fg >= 30) & (fg <= 300)
            for c, H in (("L", a), ("R", b)):
                g = self.gd_ms(H)[s2]
                out[f"gd_{c}"] = float(np.percentile(g, 90) - np.percentile(g, 10))
            # time-domain metrics of the predicted listening-position response
            irs = {}
            for c in "LR":
                hc, _ = D.chain_fir(chain_for(chain, c), self.fs, taps) if chain_for(chain, c) else (np.array([1.0]), 0)
                irs[c] = signal.fftconvolve(self.source_ir(c), hc)
            n = max(len(irs["L"]), len(irs["R"]))
            irsum = sum(np.pad(irs[c], (0, n - len(irs[c]))) for c in "LR")
            md, mx = T.modal_decay(irsum, self.an.fs)
            out["decay_med"], out["decay_max"] = md, mx
            out["c50"], out["d50"] = T.clarity(irsum, self.an.fs)
            for fc in (125, 500, 2000):
                out[f"t20_{fc}"] = T.schroeder_rt(irsum, self.an.fs, fc)
        return out


# ====================================================================== optimiser
class Options:
    def __init__(self, max_boost=3.0, max_cut=12.0, n_peq=10, f_iir=400.0, align=True, level=True,
                 speech=True, decay=True, fmin=None, fmax=None):
        self.max_boost, self.max_cut, self.n_peq, self.f_iir = max_boost, max_cut, n_peq, f_iir
        self.align, self.level, self.speech, self.decay = align, level, speech, decay
        self.fmin, self.fmax = fmin, fmax


def _weights(m, ch, o):
    fg = m.fg
    w = 1.0 / (1.0 + (m.spread[ch] / 3.0) ** 2)                  # positions disagree -> less trust
    if o.speech:
        w = w * np.where((fg >= 300) & (fg <= 4000), 1.6, 1.0)
    if o.decay:
        w = w * m.decay_w
    return w


def _band(m, ch, o):
    fg = m.fg
    fmax = min(o.fmax or m.fmax, 20000.0)
    if o.fmin:
        fmin = o.fmin
    else:                                                       # natural low-frequency roll-off: don't chase it
        e = m.base[ch] - m.tgt
        lowb = fg < 120
        k = np.where(lowb & (e > -8))[0]
        fmin = max(18.0, float(fg[k[0]]) if len(k) else 20.0)
    return (fg >= fmin) & (fg <= fmax), fmin, fmax


def _cost(e, w, sel, dip=0.35):
    ee = np.where(e > 0, e, dip * e)
    return float(np.sum((w * ee * ee)[sel]) / np.sum(w[sel]))


def _peq_db(p, m):
    fc, q, g = p
    return _biquad_db("peaking", fc, q, g, m.fs, m.fg)


def optimise(m, chain0, o, prog=lambda s: None):
    """Returns (chain, notes).  chain0 blocks are kept; the optimiser appends level, EQ and alignment blocks."""
    fg = m.fg
    chain = [dict(b) for b in chain0]
    notes = []
    # ---- 2. per-channel PEQ (IIR below f_iir, linear-phase FIR above)
    for c in "LR":
        prog(f"{'左' if c == 'L' else '右'}聲道 EQ 搜尋…")
        w = _weights(m, c, o)
        sel, fmin, fmax = _band(m, c, o)
        base = m.mag(c, chain)
        peqs = []

        def total(ps):
            y = base.copy()
            for p in ps:
                y = y + _peq_db(p, m)
            return y
        cur = _cost(total(peqs) - m.tgt, w, sel)
        c0 = cur
        done = np.zeros(len(fg), bool)          # regions where a boost already hit the cap: a dip we cannot fill
        for k in range(o.n_peq):
            e = total(peqs) - m.tgt
            es = _box_smooth(fg, e, 1 / 6)
            score = np.where(sel & ~done, w * np.where(es > 0, es, 0.35 * es) * np.abs(es), 0)
            j = int(np.argmax(np.abs(score)))
            if abs(es[j]) < 0.6:
                break
            fc0 = float(fg[j])
            half = np.abs(es) >= abs(es[j]) / 2
            lo = j
            while lo > 0 and half[lo - 1] and np.sign(es[lo - 1]) == np.sign(es[j]):
                lo -= 1
            hi = j
            while hi < len(fg) - 1 and half[hi + 1] and np.sign(es[hi + 1]) == np.sign(es[j]):
                hi += 1
            bw = max(np.log2(fg[hi] / fg[lo]), 1 / 12)
            q0 = float(np.clip(np.sqrt(2 ** bw) / (2 ** bw - 1), 0.4, 10.0))
            qmax = 12.0 if fc0 < o.f_iir else 4.0
            g0 = float(np.clip(-es[j], -o.max_cut, o.max_boost))
            bounds = [(max(fmin, fc0 / 1.4), min(fmax, fc0 * 1.4)), (0.3, qmax), (-o.max_cut, o.max_boost)]

            def fobj(p):
                return _cost(total(peqs + [p]) - m.tgt, w, sel)
            r = optimize.minimize(fobj, [fc0, min(q0, qmax), g0], bounds=bounds, method="L-BFGS-B",
                                  options=dict(maxiter=60))
            if r.fun < cur * 0.985:
                peqs.append(list(r.x))
                cur = r.fun
                if r.x[2] >= o.max_boost - 0.05:
                    done |= (fg >= r.x[0] / 2 ** 0.5) & (fg <= r.x[0] * 2 ** 0.5)
            else:
                done |= (fg >= fc0 / 2 ** 0.25) & (fg <= fc0 * 2 ** 0.25)
        if peqs:                                                # joint refinement of all filters of this channel
            x0 = np.array(peqs).ravel()
            bnd = []
            for p in peqs:
                bnd += [(max(fmin, p[0] / 1.25), min(fmax, p[0] * 1.25)), (0.3, 12.0 if p[0] < o.f_iir else 4.0),
                        (-o.max_cut, o.max_boost)]
            r = optimize.minimize(lambda x: _cost(total(list(x.reshape(-1, 3))) - m.tgt, w, sel), x0, bounds=bnd,
                                  method="L-BFGS-B", options=dict(maxiter=120))
            if r.fun <= cur:
                peqs, cur = [list(p) for p in r.x.reshape(-1, 3)], r.fun
        # merge near-duplicates (same sign, within 1/5 octave) into one filter
        merged = []
        for fc, q, g in sorted(peqs):
            if merged and abs(np.log2(fc / merged[-1][0])) < 0.2 and np.sign(g) == np.sign(merged[-1][2]):
                f0, q0, g0 = merged[-1]
                merged[-1] = [np.sqrt(f0 * fc), min(q0, q), float(np.clip(g0 + g, -o.max_cut, o.max_boost))]
            else:
                merged.append([fc, q, g])
        peqs = merged
        for fc, q, g in sorted(peqs):
            if abs(g) < 0.3:
                continue
            chain.append(dict(type="peq", shape="peaking", fc=round(float(fc), 1), q=round(float(q), 2),
                              gain=round(float(g), 1), phase="natural" if fc < o.f_iir else "linear", ch=c, auto=True))
        notes.append(f"{'左' if c == 'L' else '右'}聲道：{len([p for p in peqs if abs(p[2]) >= 0.3])} 個 PEQ，"
                     f"範圍 {fmin:.0f} Hz–{fmax / 1000:g} kHz，加權誤差 {np.sqrt(c0):.2f} → {np.sqrt(cur):.2f} dB")
    # ---- 2b. channel balance after EQ (cut the louder channel only)
    if o.level:
        mid = (fg >= 500) & (fg <= 2000)
        d = float(np.mean(m.mag("L", chain)[mid] - m.mag("R", chain)[mid]))
        if abs(d) >= 0.2:
            loud = "L" if d > 0 else "R"
            chain.append(dict(type="gain", db=-round(abs(d), 1), ch=loud, auto=True))
            notes.append(f"左右音量：{'左' if loud == 'L' else '右'}聲道降 {abs(d):.1f} dB（500 Hz–2 kHz 對齊）")
    # ---- 3. L/R phase: delay + up to two all-pass on one channel, maximise the coherent sum
    if o.align and m.has_center:
        prog("左右相位對齊搜尋…")
        s_lo = (fg >= 30) & (fg <= 1000)
        s_hi = (fg >= 1000) & (fg <= 6000)
        A0, B0 = m.center_resp("L", chain), m.center_resp("R", chain)
        wl = 1.0 / fg[s_lo]

        def coh(a, b, s, wt=None):
            wt = np.ones(s.sum()) if wt is None else wt
            return float(np.sum(wt * np.abs(a + b)[s] ** 2) / np.sum(wt * (np.abs(a) + np.abs(b))[s] ** 2))
        gd0 = 0.5 * (np.median(m.gd_ms(A0)[(fg >= 30) & (fg <= 200)]) + np.median(m.gd_ms(B0)[(fg >= 30) & (fg <= 200)]))

        def build(x):
            dms, f1, q1, f2, q2, on1, on2, which = x
            blocks = []
            if abs(dms) >= 0.01:
                blocks.append(dict(type="delay", ms=round(abs(dms), 3), ch="R" if dms > 0 else "L", auto=True))
            tgt_ch = "L" if which < 0.5 else "R"
            if on1 > 0.5:
                blocks.append(dict(type="allpass", fc=round(f1, 1), q=round(q1, 2), ch=tgt_ch, auto=True))
            if on2 > 0.5:
                blocks.append(dict(type="allpass", fc=round(f2, 1), q=round(q2, 2), ch=tgt_ch, auto=True))
            return blocks

        def fobj(x):
            bl = build(x)
            a = A0 * m.chain_H(chain_for(bl, "L"))
            b = B0 * m.chain_H(chain_for(bl, "R"))
            gd = 0.5 * (np.median(m.gd_ms(a)[(fg >= 30) & (fg <= 200)]) + np.median(m.gd_ms(b)[(fg >= 30) & (fg <= 200)]))
            pen = 0.004 * max(gd - gd0, 0.0) + 0.02 * (x[5] > 0.5) + 0.02 * (x[6] > 0.5)
            return -(coh(a, b, s_lo, wl) + 0.3 * coh(a, b, s_hi)) + pen
        bnds = [(-1.5, 1.5), (30, 500), (0.3, 1.2), (30, 500), (0.3, 1.2), (0, 1), (0, 1), (0, 1)]
        r = optimize.differential_evolution(fobj, bnds, popsize=14, maxiter=60, tol=1e-4, seed=1, polish=False)
        base_score = fobj(np.array([0, 100, 1, 100, 1, 0, 0, 0]))
        c_lo0 = coh(A0, B0, s_lo, wl)
        if r.fun < base_score - 0.01:
            bl = build(r.x)
            if any(b_["type"] == "allpass" for b_ in bl) and o.decay:
                # all-pass sections ring: keep them only if the predicted modal decay does not get longer
                d0 = m.metrics(chain, 16384).get("decay_max", 0)
                d1 = m.metrics(chain + bl, 16384).get("decay_max", 0)
                if d1 > d0 * 1.08:
                    bl = [b_ for b_ in bl if b_["type"] == "delay"]
                    notes.append(f"全通會拉長低頻餘音（{d0:.0f} → {d1:.0f} ms），已略過，只保留延遲")
            a = A0 * m.chain_H(chain_for(bl, "L"))
            b = B0 * m.chain_H(chain_for(bl, "R"))
            chain += bl
            desc = []
            for b_ in bl:
                if b_["type"] == "delay":
                    desc.append(f"延遲{'右' if b_['ch'] == 'R' else '左'}聲道 {b_['ms']:.2f} ms")
                else:
                    desc.append(f"{'左' if b_['ch'] == 'L' else '右'}聲道全通 {b_['fc']:.0f} Hz Q{b_['q']:.2f}")
            notes.append("左右相位：" + "、".join(desc) + f"；30 Hz–1 kHz 合成一致性 {c_lo0 * 100:.0f}% → {coh(a, b, s_lo, wl) * 100:.0f}%")
        else:
            notes.append(f"左右相位：目前已接近最佳（一致性 {c_lo0 * 100:.0f}%），不加延遲或全通")
    return chain, notes


def metrics_table(before, after):
    rows = [("左 RMS 偏差 20–200 Hz", "rms_lo_L", "dB"), ("左 RMS 偏差 200–2k", "rms_mid_L", "dB"),
            ("左 RMS 偏差 2k–20k", "rms_hi_L", "dB"),
            ("右 RMS 偏差 20–200 Hz", "rms_lo_R", "dB"), ("右 RMS 偏差 200–2k", "rms_mid_R", "dB"),
            ("右 RMS 偏差 2k–20k", "rms_hi_R", "dB"),
            ("左右頻響差 20–200 Hz", "lr_lo", "dB"), ("左右頻響差 200–2k", "lr_mid", "dB"), ("左右頻響差 2k–20k", "lr_hi", "dB"),
            ("左右音量差 500–2k（左−右）", "bal", "dB"),
            ("L+R 合成一致性 40–400 Hz", "coh", "%"),
            ("群延遲起伏 30–300 Hz（左）", "gd_L", "ms"), ("群延遲起伏 30–300 Hz（右）", "gd_R", "ms"),
            ("低頻衰減 30–250 Hz 降 20 dB（中位）", "decay_med", "ms"), ("低頻衰減（最長）", "decay_max", "ms"),
            ("RT（T20）125 Hz", "t20_125", "s"), ("RT（T20）500 Hz", "t20_500", "s"), ("RT（T20）2 kHz", "t20_2000", "s"),
            ("清晰度 C50", "c50", "dB"), ("清晰度 D50", "d50", "%")]
    out = []
    for name, k, u in rows:
        if k not in before or k not in after:
            continue
        a, b = before[k], after[k]
        if a != a or b != b:
            continue
        fmt = "{:.2f}" if u == "s" else ("{:.0f}" if u in ("ms", "%") else "{:+.1f}" if k == "bal" else "{:.1f}")
        out.append(f"{name:<24}{fmt.format(a):>8} → {fmt.format(b):<8}{u}")
    return out
