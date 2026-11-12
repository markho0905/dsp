"""roomcorr_analysis.py - analyse a measurement session and (re)design filters for any target curve."""
from types import SimpleNamespace

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

import roomcorr as R

TINY = 1e-12

DEFAULTS = dict(taps=16384, fmin=25, fmax=5000, max_boost=6.0, max_cut=15.0, phase_fmax=400,
                delay_ms=30.0, headroom=1.0, cycles=15, tmax=0.5, ref_band=(300, 3000),
                target_file=None, target_points=None, bass=0, bass_f=100, tilt=0, time_align=False)


def make_params(**kw):
    d = dict(DEFAULTS)
    d.update(kw)
    return SimpleNamespace(**d)


FLAT_PTS = [(20, 0), (100, 0), (1000, 0), (10000, 0), (20000, 0)]
PRESET_POINTS = {
    "平坦": FLAT_PTS,
    "低頻增強": [(20, 5), (50, 5), (100, 3), (200, 1), (500, 0), (1000, 0), (20000, 0)],
    "高頻柔和": [(20, 0), (1000, 0), (3000, -0.5), (6000, -1.5), (10000, -2.5), (20000, -4)],
}
SLOTS = [("std", "標準"), ("night", "夜間"), ("movie", "電影")]
BASE_SETTINGS = dict(preset="平坦", bass=0.0, knee=100.0, tilt=0.0, fmax=5000.0, fmin=25.0, points=None,
                     phase="mixed", taps=16384, out_fs=48000, smooth=1 / 6, max_boost=6.0, max_cut=15.0,
                     mimo=False, decay=False, limit_boost=True, delay_ms=30.0, phase_fmax=400.0, dip=10.0,
                     pre_ms=30.0, align=False, headroom=1.0, keep_bass=False, mimo_fmax=150.0, mimo_support=1.0,
                     mimo_mode="space")
SLOT_DEFAULTS = {"std": {}, "night": dict(preset="自訂", bass=-3.0, knee=100.0, tilt=-1.0),
                 "movie": dict(preset="低頻增強")}


def slot_settings(slot, saved=None):
    d = dict(BASE_SETTINGS)
    d.update(SLOT_DEFAULTS.get(slot, {}))
    if saved:
        d.update(saved)
    return d


def target_points_from(s):
    """Target curve points for a settings dict (preset, sliders or hand-dragged points)."""
    if s.get("points"):
        return [tuple(p) for p in s["points"]]
    if s["preset"] in PRESET_POINTS:
        return list(PRESET_POINTS[s["preset"]])
    fg = np.geomspace(20, 20000, 14)
    y = s["bass"] / (1 + (fg / max(s["knee"], 1.0)) ** 2) + s["tilt"] * np.clip(np.log10(fg / 2000.0), 0, 1)
    return [(float(a), float(b)) for a, b in zip(fg, y)]


def params_from_settings(s):
    mode = s["phase"]
    return make_params(taps=int(s["taps"]), fmin=float(s.get("fmin", 25.0)), fmax=float(s["fmax"]),
                       max_boost=float(s["max_boost"]) if s["limit_boost"] else 12.0, max_cut=float(s["max_cut"]),
                       phase_fmax=0.0 if mode == "min" else float(s["phase_fmax"]), phase_mode=mode,
                       delay_ms=float(s["delay_ms"]), headroom=float(s["headroom"]), dip_limit=float(s["dip"]),
                       pre_ms=float(s["pre_ms"]) if s["pre_ms"] < s["delay_ms"] else None,
                       time_align=bool(s["align"]), target_points=target_points_from(s),
                       smooth_oct=float(s["smooth"]), decay_control=bool(s["decay"]), mimo=bool(s["mimo"]),
                       keep_bass=bool(s.get("keep_bass", False)), mimo_fmax=float(s.get("mimo_fmax", 150.0)),
                       mimo_support=float(s.get("mimo_support", 1.0)),
                       mimo_phase=1.0 if s.get("mimo_mode", "space") == "decay" else 0.0)


def rt60(ir, fs):
    """Schroeder backward integration, T20 extrapolated to 60 dB.  None if not measurable."""
    p = int(np.argmax(np.abs(ir)))
    e = ir[p:] ** 2
    n = len(e)
    if n < 2000:
        return None
    e = np.maximum(e - np.mean(e[int(0.85 * n):]), 1e-20)
    edc = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(edc / edc[0] + 1e-20)
    i1, i2 = int(np.argmax(db <= -5)), int(np.argmax(db <= -25))
    if i2 <= i1 + 20 or db[i2] > -24:
        return None
    slope = np.polyfit(np.arange(i1, i2) / fs, db[i1:i2], 1)[0]
    return -60.0 / slope if slope < 0 else None


def rt60_band(ir, fs, fc):
    sos = signal.butter(4, [fc / 2 ** 0.5, fc * 2 ** 0.5], "bp", fs=fs, output="sos")
    return rt60(signal.sosfiltfilt(sos, ir), fs)


class Analysis:
    def __init__(self, meas, both, fs, mic_cal=None, nfft=1 << 18, progress=None):
        self.fs, self.nfft = fs, nfft
        self.f = rfftfreq(nfft, 1 / fs)
        self.base = make_params()
        self.cal = R.load_curve(mic_cal) if mic_cal else None
        self.meas, self.both = meas, both
        self.data = {}
        for ch in ("L", "R"):
            if progress:
                progress(f"分析{'左' if ch == 'L' else '右'}喇叭…")
            pos = sorted(meas[ch])
            f, Hs = R.analyse([meas[ch][p] for p in pos], fs, nfft, self.base, self.cal)
            d = R.prep(f, Hs, self.base)
            d["pos"] = pos
            self.data[ch] = d
        self.common = float(np.mean([self.data[c]["ref"] for c in "LR"]))
        self.w = R.smooth_width(self.f)
        self.ix = np.unique(np.geomspace(2, len(self.f) - 1, 1200).astype(int))
        self._center()
        self.report = self._report()

    # ---- complex spectra per position with preserved L/R timing
    def _center(self):
        self.pairs, self.center = {}, None
        fs, b = self.fs, self.base
        pre = int(0.005 * fs)
        Lw = pre + int((b.tmax + 0.05) * fs)
        for pos in sorted(set(self.meas["L"]) & set(self.meas["R"])):
            irL, irR = self.meas["L"][pos], self.meas["R"][pos]
            pkL, pkR = int(np.argmax(np.abs(irL))), int(np.argmax(np.abs(irR)))
            start = min(pkL, pkR) - pre

            def sp(ir, start=start):
                H = R.fdw_spectrum(R.crop(ir, start, Lw), pre, fs, self.nfft, b.cycles, b.tmax)
                if self.cal is not None:
                    H = H / (10 ** (R.curve_at(self.f, self.cal) / 20))
                return H
            c = dict(pos=pos, dT=pkR - pkL, HL=sp(irL), HR=sp(irR), tau=pre / fs)
            if pos in self.both:
                c["HB"] = sp(self.both[pos])
            self.pairs[pos] = c
        if self.pairs:
            self.center = self.pairs.get(1) or self.pairs[min(self.pairs)]

    def sm(self, H):
        return 10 * np.log10(R.oct_smooth(np.abs(H) ** 2, self.w) + TINY)

    # ---- static report (independent of target)
    def _report(self):
        f, fs = self.f, self.fs
        L, R_ = self.data["L"], self.data["R"]
        out = []
        out.append(f"取樣率 {fs} Hz；位置數：左 {len(L['pos'])}、右 {len(R_['pos'])}"
                   f"{'；有左右合併量測' if self.both else '；無左右合併量測'}")
        if min(len(L["pos"]), len(R_["pos"])) < 3:
            out.append("⚠ 位置少於 3 個，修正結果較不穩定。")
        sel = (f >= 200) & (f <= 2000)
        bal = float(np.mean(R_["M"][sel] - L["M"][sel]))
        out.append(f"左右平衡（200–2000 Hz）：右喇叭比左喇叭{'大聲' if bal > 0 else '小聲'} {abs(bal):.1f} dB"
                   + ("　→ 建議在擺位或放大器端調整" if abs(bal) > 1.0 else "（良好）"))
        if self.center:
            dT = self.center["dT"]
            ms = dT / fs * 1000
            cm = dT / fs * 343 * 100
            out.append(f"到達時間差（位置 {self.center['pos']}）：右比左{'晚' if dT > 0 else '早'} {abs(ms):.2f} ms"
                       f"（距離差約 {abs(cm):.0f} cm）" + ("；可勾選「左右時間對齊」補償" if abs(ms) > 0.1 else ""))
        lg = np.geomspace(25, 300, 300)
        for ch, d in (("左", L), ("右", R_)):
            dev = np.interp(lg, f, d["M"] - self.common)
            pk, _ = signal.find_peaks(dev, prominence=3)
            pk = [i for i in pk if dev[i] > 1.0]
            dp, _ = signal.find_peaks(-dev, prominence=3)
            dp = [i for i in dp if dev[i] < -1.0]
            pk = sorted(pk, key=lambda i: -dev[i])[:3]
            dp = sorted(dp, key=lambda i: dev[i])[:2]
            s = "、".join([f"峰 {lg[i]:.0f} Hz ({dev[i]:+.1f} dB)" for i in pk] +
                         [f"谷 {lg[i]:.0f} Hz ({dev[i]:+.1f} dB)" for i in dp])
            out.append(f"{ch}喇叭低頻（25–300 Hz，相對中頻）：{s or '無明顯峰谷'}")
        for ch, d in (("左", L), ("右", R_)):
            if len(d["pos"]) > 1:
                db = np.array([self.sm(H) for H in d["Hs"]])
                sp = db.std(axis=0)
                lo = sp[(f >= 25) & (f <= 300)].mean()
                hi = sp[(f >= 300) & (f <= 3000)].mean()
                out.append(f"{ch}喇叭位置間差異：25–300 Hz {lo:.1f} dB，300–3k Hz {hi:.1f} dB"
                           "（越大代表越難用單一濾波器修正）")
        if self.center:
            ir = self.meas["L"][self.center["pos"]]
            parts = []
            for lab, v in (("全頻", rt60(ir, fs)), ("125 Hz", rt60_band(ir, fs, 125)),
                           ("500 Hz", rt60_band(ir, fs, 500)), ("2 kHz", rt60_band(ir, fs, 2000))):
                parts.append(f"{lab} {v:.2f} s" if v else f"{lab} 無法估計")
            out.append("殘響時間 RT60（粗估）：" + "；".join(parts))
            if "HB" in self.center:
                c = self.center
                band = (f >= 30) & (f <= 5000)
                err = np.sqrt(np.mean((self.sm(c["HB"]) - self.sm(c["HL"] + c["HR"]))[band] ** 2))
                out.append(f"合併量測 vs 左右單獨相加：差 {err:.1f} dB（RMS）"
                           + ("，時序一致" if err < 2.5 else "，偏大：可能是裝置延遲不穩或環境噪音，建議重測"))
        return out

    # ---- filter design for the current target / parameters
    def _mimo(self, p, hs, info):
        """MIMO joint correction with support loudspeakers (Dirac ART style), below p.mimo_fmax (default 150 Hz).

        For each source channel (L programme, R programme) the primary speaker keeps its SISO room-correction
        filter and the *other* speaker gets a support filter.  Both are solved together, per frequency bin, by least
        squares over every position where L and R were measured:

            sum_m | H_m,p (F_p + w dP) + H_m,s (w S) - D_m |^2  + beta |dP|^2 + (beta + mu) |S|^2

        H_m,p / H_m,s: measured responses of the primary / support speaker at position m (same time origin),
        F_p: SISO filter, D_m: the target magnitude with the phase of the primary's own direct sound at that
        position (so stereo timing is kept and the support only cancels what the room adds), w: band weight that
        fades the whole MIMO contribution out above mimo_fmax.  mu grows as mimo_support shrinks (0..1).
        p.mimo_phase (from the mimo_mode setting): 0 = "space" - the target keeps the phase the SISO filter already
        produces, so the support only evens out the level between positions (safe, small headroom cost);
        1 = "decay" - the target has the primary's direct-sound phase, so the support also cancels the room's
        modal decay (Dirac ART behaviour; costs more headroom and may leave common dips deeper).
        Returns (filters per channel, cross filters {"LR": source L -> speaker R, "RL": ...}, used)."""
        prs = [self.pairs[k] for k in sorted(self.pairs)]
        if len(prs) < 2:
            return hs, {}, False
        fs, f, nfft = self.fs, self.f, self.nfft
        fm = float(np.clip(getattr(p, "mimo_fmax", 150.0), 40.0, min(float(p.fmax), 300.0)))
        k1 = int(np.searchsorted(f, fm * 2.0)) + 1
        fk = f[:k1]
        w = R.band_weight(fk, p.fmin, fm, 0.5, 0.75)
        lat = (p.taps // 2 / fs) if getattr(p, "phase_mode", "mixed") == "linear" else p.delay_ms * 1e-3
        strength = float(np.clip(getattr(p, "mimo_support", 1.0), 0.0, 1.0))
        H = {c: np.stack([q["H" + c][:k1] for q in prs]) for c in "LR"}              # M x K
        F = {c: rfft(hs[c], nfft) for c in "LR"}
        energy = np.mean(np.abs(H["L"]) ** 2 + np.abs(H["R"]) ** 2, axis=0) / 2 + TINY
        beta = float(getattr(p, "mimo_beta", 0.05)) * energy
        mu = beta * 20.0 * ((1.0 - strength) / max(strength, 0.05)) ** 2 if strength < 1.0 else 0.0
        if strength <= 0.0:
            mu = beta * 1e6
        band = (fk >= p.fmin) & (fk <= fm)
        mid = (f >= 300) & (f <= 3000)
        out_h, out_x = {}, {}
        for src in "LR":
            oth = "R" if src == "L" else "L"
            Fp = F[src][:k1]
            own = H[src] * Fp[None, :]                                           # what the SISO filter achieves
            # direct-sound arrival of the primary speaker at each position (inside the pair window)
            tau = np.array([q["tau"] + max(0, (-q["dT"] if src == "L" else q["dT"])) / fs for q in prs])
            # target magnitude: the curve target, but never more than ~3 dB above what the SISO design already
            # reaches on average (dips it chose not to fill stay unfilled; per-position nulls can still be filled)
            avg_db = 10 * np.log10(R.oct_smooth(np.mean(np.abs(own) ** 2, axis=0), 1 / 6) + TINY)
            amp = 10 ** (np.minimum(info[src]["tgt"][:k1], avg_db + 3.0) / 20)
            # target phase: the primary's own direct sound inside the MIMO band, blending into the phase the SISO
            # filter already produces where the band fades out (no complex blend of two unrelated solutions)
            u_direct = np.exp(-2j * np.pi * fk[None, :] * (tau[:, None] + lat))
            u_own = own / (np.abs(own) + TINY)
            rho = float(np.clip(getattr(p, "mimo_phase", 1.0), 0.0, 1.0))   # 1 = direct-sound phase (decay control)
            u = (w * rho)[None, :] * u_direct + (1 - w * rho)[None, :] * u_own
            u = u / (np.abs(u) + TINY)
            D = amp[None, :] * u
            resid = D - own
            Ap, As = w[None, :] * H[src], w[None, :] * H[oth]
            a11_0 = np.sum(np.abs(Ap) ** 2, axis=0)
            a22_0 = np.sum(np.abs(As) ** 2, axis=0)
            a12 = np.sum(np.conj(Ap) * As, axis=0)
            b1 = np.sum(np.conj(Ap) * resid, axis=0)
            b2 = np.sum(np.conj(As) * resid, axis=0)
            # gain limits relative to the SISO filter: primary up to 2 dB over its own peak, support up to that peak.
            # Instead of clipping the solution (which breaks the cancellation pattern), the regularisation of a bin
            # is raised until its solution respects the limits, so it shrinks smoothly back towards the SISO filter.
            g_mid = float(np.median(np.abs(F[src][mid]))) + TINY
            pk = max(float(np.abs(Fp[band]).max()) if band.any() else g_mid, g_mid * 10 ** (p.max_boost / 20))
            lim_p, lim_s = pk * 10 ** (2.0 / 20), pk
            prim_pen = 0.0 if getattr(p, "mimo_primary", True) else beta * 1e6
            scale_k = np.ones(k1)
            for _ in range(24):
                a11 = a11_0 + beta * scale_k + prim_pen
                a22 = a22_0 + (beta + mu) * scale_k
                det = a11 * a22 - np.abs(a12) ** 2 + TINY
                dP = (a22 * b1 - a12 * b2) / det
                S = (a11 * b2 - np.conj(a12) * b1) / det
                Wp, Ws = Fp + w * dP, w * S
                bad = (np.abs(Wp) > lim_p) | (np.abs(Ws) > lim_s)
                if not bad.any():
                    break
                scale_k = np.where(bad, scale_k * 2.0, scale_k)
            Wp = Wp * np.minimum(1.0, lim_p / (np.abs(Wp) + TINY))        # safety only; normally inactive now
            Ws = Ws * np.minimum(1.0, lim_s / (np.abs(Ws) + TINY))
            k = p.taps // 5
            fade = 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
            Fn = F[src].copy()
            Fn[:k1] = Wp
            h = np.fft.irfft(Fn, nfft)[:p.taps]
            h[-k:] *= fade
            out_h[src] = h
            Fx = np.zeros_like(Fn)
            Fx[:k1] = Ws
            hx = np.fft.irfft(Fx, nfft)[:p.taps]
            hx[-k:] *= fade
            out_x[src + oth] = hx
        self.mimo_band = (float(p.fmin), fm)
        return out_h, out_x, True

    def _achieved(self, pos, Ft, Fx):
        """Complex response at one measured position for source L, source R and L+R programme (all paths)."""
        q = self.pairs.get(pos)
        if q is None:
            return None
        xLR = Fx.get("LR", 0.0)
        xRL = Fx.get("RL", 0.0)
        aL = q["HL"] * Ft["L"] + q["HR"] * xLR
        aR = q["HR"] * Ft["R"] + q["HL"] * xRL
        return {"L": aL, "R": aR, "LR": aL + aR}

    @staticmethod
    def _band_stats(f, w, before, after, tgt, lo, hi):
        """Same numbers as roomcorr.band_report, from per-position complex spectra (M x K) before/after."""
        sel = (f >= lo) & (f <= hi)

        def stat(Hs):
            db = np.array([10 * np.log10(R.oct_smooth(np.abs(H) ** 2, w) + TINY) for H in Hs])
            avg = 10 * np.log10(R.oct_smooth(np.mean(np.abs(Hs) ** 2, axis=0), w) + TINY)
            e_avg = np.sqrt(np.mean((avg - tgt)[sel] ** 2))
            e_pos = np.mean([np.sqrt(np.mean((x - tgt)[sel] ** 2)) for x in db])
            return e_avg, e_pos
        return stat(before), stat(after)

    def curves_for(self, ds, kind, pos):
        """(before, after) in dB relative to the mid-band level for one cell of the measurement matrix."""
        f = self.f
        Fx = getattr(ds, "Fx", {}) or {}
        if kind in ("L", "R"):
            d = self.data[kind]
            if pos not in d["pos"]:
                return None
            H = d["Hs"][d["pos"].index(pos)]
            b = self.sm(H)
            ach = self._achieved(pos, ds.Ft, Fx) if Fx else None
            a = self.sm(ach[kind]) if ach is not None else self.sm(H * ds.Ft[kind])
            ref = self.common
        else:
            q = self.pairs.get(pos)
            if q is None:
                return None
            b = self.sm(q["HB"] if "HB" in q else q["HL"] + q["HR"])
            a = self.sm(self._achieved(pos, ds.Ft, Fx)["LR"])
            ref = float(b[(f >= 300) & (f <= 3000)].mean())
            a = a - float(a[(f >= 300) & (f <= 3000)].mean()) + 0.0
        return b - ref, a - ref if kind in ("L", "R") else a

    def design(self, p):
        fs, f, nfft = self.fs, self.f, self.nfft
        hs, info, Ft, hx, Fx = {}, {}, {}, {}, {}
        for ch in "LR":
            hs[ch], info[ch] = R.design_filter(f, self.data[ch], fs, nfft, p, self.common)
        mimo_used = False
        if getattr(p, "mimo", False):
            hs, hx, mimo_used = self._mimo(p, hs, info)
        shift = {"L": 0, "R": 0}
        if p.time_align and self.center:
            dT = self.center["dT"]
            if dT > 0:
                shift["L"] = dT
            elif dT < 0:
                shift["R"] = -dT
        for ch in "LR":
            if shift[ch]:
                hs[ch] = np.concatenate([np.zeros(shift[ch]), hs[ch]])[:p.taps]
            Ft[ch] = rfft(hs[ch], nfft)
        for key in list(hx):                     # cross filters carry the time alignment of their *source* channel
            if shift[key[0]]:
                hx[key] = np.concatenate([np.zeros(shift[key[0]]), hx[key]])[:p.taps]
            Fx[key] = rfft(hx[key], nfft)
        band = (f >= 20) & (f <= 20000)
        peak = {}
        for ch in "LR":                          # per speaker output: own path alone, or own + cross path for mono content
            oth = "R" if ch == "L" else "L"
            x = Fx[oth + ch][band] if (oth + ch) in Fx else 0.0
            peak[ch] = np.maximum(np.abs(Ft[ch][band]), np.abs(Ft[ch][band] + x))
        gmax = max(float(peak[c].max()) for c in "LR")
        scale = min(1.0, 10 ** (-p.headroom / 20) / gmax)
        tgt_rel = R.target_db(f, p)
        per = {}
        prs = sorted(self.pairs)
        for ch in "LR":
            d = self.data[ch]
            if mimo_used and prs:
                before = np.stack([self.pairs[q]["H" + ch] for q in prs])
                after_c = np.stack([self._achieved(q, Ft, Fx)[ch] for q in prs])
                after = 10 * np.log10(R.oct_smooth(np.mean(np.abs(after_c) ** 2, axis=0), d["w"]) + TINY)
                st_lo = self._band_stats(f, d["w"], before, after_c, info[ch]["tgt"], p.fmin, 300)
                st_hi = self._band_stats(f, d["w"], before, after_c, info[ch]["tgt"], 300, p.fmax)
            else:
                after = 10 * np.log10(R.oct_smooth(np.mean(np.abs(d["Hs"]) ** 2, axis=0) * np.abs(Ft[ch]) ** 2, d["w"]) + TINY)
                st_lo = R.band_report(f, d, Ft[ch], info[ch]["tgt"], p.fmin, 300)
                st_hi = R.band_report(f, d, Ft[ch], info[ch]["tgt"], 300, p.fmax)
            per[ch] = dict(before=d["M"] - self.common, after=after - self.common,
                           stats_lo=st_lo, stats_hi=st_hi, loss=info[ch]["loss"])
        comb = None
        if self.center:
            c = self.center
            before = self.sm(c["HL"] + c["HR"])
            sel = (f >= p.ref_band[0]) & (f <= p.ref_band[1])
            ref_c = float(before[sel].mean())
            comb = dict(before=before - ref_c, after=self.sm(self._achieved(c["pos"], Ft, Fx)["LR"]) - ref_c)
            comb["after"] = comb["after"] - (comb["after"][sel].mean())      # show relative to its own mid-band level
            if "HB" in c:
                comb["measured"] = self.sm(c["HB"]) - float(self.sm(c["HB"])[sel].mean())
        mimo_stats = self._mimo_stats(p, Ft, Fx) if mimo_used else None
        return SimpleNamespace(h={c: hs[c] * scale for c in "LR"}, hx={k: v * scale for k, v in hx.items()},
                               scale=scale, shift=shift, per=per, comb=comb, tgt_rel=tgt_rel, params=p, Ft=Ft, Fx=Fx,
                               mimo_used=mimo_used, mimo_stats=mimo_stats, info=info)

    def _mimo_stats(self, p, Ft, Fx):
        """Spread across positions (dB) in the MIMO band and predicted modal decay at the centre: SISO-only part vs
        with support paths.  Used for the design report."""
        f = self.f
        lo, hi = getattr(self, "mimo_band", (p.fmin, 150.0))
        sel = (f >= lo) & (f <= hi)
        prs = sorted(self.pairs)
        if len(prs) < 2:
            return None
        out = {}
        for src in "LR":
            own = np.stack([self.pairs[q]["H" + src] * Ft[src] for q in prs])
            full = np.stack([self._achieved(q, Ft, Fx)[src] for q in prs])

            def spread(Hs):
                db = np.array([10 * np.log10(R.oct_smooth(np.abs(H) ** 2, 1 / 6) + TINY) for H in Hs])
                return float(db[:, sel].std(axis=0).mean())
            out["spread_" + src] = (spread(own), spread(full))
        sup = {k: 20 * np.log10(np.abs(v[sel]).max() / (np.abs(Ft[k[0]][sel]).max() + TINY) + TINY) for k, v in Fx.items()}
        out["support_db"] = sup
        return out

    def design_text(self, ds):
        p = ds.params
        out = []
        for ch, name in (("L", "左"), ("R", "右")):
            r = ds.per[ch]
            (b1, bp1), (a1, ap1) = r["stats_lo"]
            (b2, bp2), (a2, ap2) = r["stats_hi"]
            out.append(f"{name}喇叭 與目標的 RMS 誤差：{p.fmin:.0f}–300 Hz {b1:.1f} → {a1:.1f} dB；"
                       f"300–{p.fmax:.0f} Hz {b2:.1f} → {a2:.1f} dB（預測，平均響應）")
        lat = p.taps / 2 / self.fs * 1000 if getattr(p, "phase_mode", "mixed") == "linear" else p.delay_ms
        bo = [ds.info[c].get("bass_off", 0.0) for c in "LR"]
        if getattr(p, "keep_bass", False):
            out.append(f"低頻音量基準：保留原本低頻能量（目標在 300 Hz 以下平移 左 {bo[0]:+.1f}、右 {bo[1]:+.1f} dB，"
                       "削峰與補谷在低頻互相抵銷，整體低音量不變）")
        if getattr(ds, "mimo_used", False):
            lo, hi = getattr(self, "mimo_band", (p.fmin, 150.0))
            ms = getattr(ds, "mimo_stats", None) or {}
            sp = [ms.get("spread_" + c) for c in "LR"]
            sup = ms.get("support_db", {})
            mode = "餘音控制" if getattr(p, "mimo_phase", 0.0) >= 0.5 else "空間一致"
            txt = (f"MIMO 支援喇叭（{lo:.0f}–{hi:.0f} Hz，{mode}）：左聲道由右喇叭支援（最大 {sup.get('LR', 0):+.0f} dB）、"
                   f"右聲道由左喇叭支援（最大 {sup.get('RL', 0):+.0f} dB）")
            if all(sp):
                txt += f"；位置間差異 左 {sp[0][0]:.1f} → {sp[0][1]:.1f} dB、右 {sp[1][0]:.1f} → {sp[1][1]:.1f} dB（只用自己的濾波器 → 加上支援）"
            out.append(txt)
        out.append(f"濾波器整體增益 {20 * np.log10(ds.scale):.1f} dB；延遲約 {lat:.0f} ms"
                   + ("；已啟用 MIMO 聯合校正" if getattr(ds, "mimo_used", False) else "")
                   + (f"；時間對齊：延遲{'左' if ds.shift['L'] else '右'}聲道 "
                      f"{max(ds.shift.values()) / self.fs * 1000:.2f} ms" if any(ds.shift.values()) else ""))
        return out


# ====================================================================== auto optimisation
AUTO_TARGET = [(20.0, 0.0), (200.0, 0.0), (1000.0, 0.0), (10000.0, -1.5), (20000.0, -3.0)]


def _band_rms(f, y, lo, hi):
    sel = (f >= lo) & (f <= hi)
    return float(np.sqrt(np.mean(y[sel] ** 2))) if sel.any() else 0.0


def auto_settings(an, settings, conservative=0.5, progress=None):
    """Derive filter settings from the measured data and pick the best of a few candidates.
    Returns (new_settings, summary_lines).  conservative 0..1: 0 = correct as much as possible, 1 = gentle."""
    f = an.f
    s = dict(settings)
    L, Rr = an.data["L"], an.data["R"]
    lines = []
    # --- how low do the speakers go?  lowest f where both channels stay within 10 dB of mid-band
    fmin = 25.0
    for fc in np.geomspace(20, 80, 40):
        sel = (f >= fc / 1.1) & (f <= fc * 1.1)
        ok = all(np.mean(d["M"][sel] - an.common) > -10 for d in (L, Rr))
        if ok:
            fmin = max(20.0, float(fc) * 1.1)
            break
        fmin = float(fc) * 1.1
    fmin = float(np.clip(round(fmin / 5) * 5, 20, 60))
    # --- is the HF data usable?  average level and spread between positions in 8-16 kHz
    hf_ok = True
    for d in (L, Rr):
        sel = (f >= 8000) & (f <= 16000)
        if np.mean(d["M"][sel] - an.common) < -18:
            hf_ok = False
        if len(d["pos"]) > 1:
            db = np.array([10 * np.log10(R.oct_smooth(np.abs(H) ** 2, 1 / 3) + 1e-12) for H in d["Hs"]])
            if db[:, sel].std(axis=0).mean() > 6:
                hf_ok = False
    fmax = 16000.0 if hf_ok else 5000.0
    # --- consistency between positions at LF decides how hard we may cut
    spread = []
    for d in (L, Rr):
        if len(d["pos"]) > 1:
            db = np.array([10 * np.log10(R.oct_smooth(np.abs(H) ** 2, 1 / 6) + 1e-12) for H in d["Hs"]])
            sel = (f >= fmin) & (f <= 300)
            spread.append(float(db[:, sel].std(axis=0).mean()))
    spread = float(np.mean(spread)) if spread else 4.0
    npos = min(len(L["pos"]), len(Rr["pos"]))
    # --- time alignment if the arrival difference is clearly measurable
    align = bool(an.center and abs(an.center["dT"]) / an.fs * 1000 > 0.15)
    boost = float(np.interp(conservative, [0, 1], [8, 4]))
    s.update(fmin=fmin, fmax=fmax, max_boost=round(boost, 1), limit_boost=True, dip=10.0, phase="mixed",
             phase_fmax=400.0, decay=True, mimo=bool(settings.get("mimo", False)), align=align, bass=0.0, tilt=0.0,
             knee=100.0)
    if s.get("preset") in ("平坦", "自訂") and not s.get("points"):
        s["preset"], s["points"] = "自訂", [list(p) for p in AUTO_TARGET]
    # --- candidate search: how much to cut, how fine to follow the curve
    cuts = [9.0, 12.0, 15.0] if spread < 5 else [6.0, 9.0, 12.0]
    smooths = [1 / 6, 1 / 12] if npos >= 3 else [1 / 6, 1 / 3]
    best = None
    k, n = 0, len(cuts) * len(smooths)
    for cut in cuts:
        for sm in smooths:
            k += 1
            if progress:
                progress(f"自動最佳化：試算 {k}/{n}…")
            cand = dict(s, max_cut=cut, smooth=sm)
            ds = an.design(params_from_settings(cand))
            cost, used = 0.0, 0.0
            for ch in "LR":
                r = ds.per[ch]
                tgt = ds.tgt_rel
                e_lo = _band_rms(f, r["after"] - tgt, fmin, 300)
                e_hi = _band_rms(f, r["after"] - tgt, 300, fmax)
                (_, _), (_, pos_lo) = r["stats_lo"]
                cost += 0.6 * e_lo + 0.4 * pos_lo + 0.5 * e_hi
                used += float(np.mean(np.abs(ds.info[ch]["corr"]))) / 10
            cost += conservative * used
            if best is None or cost < best[0]:
                best = (cost, cand, ds)
    cost, cand, ds = best
    base = an.design(params_from_settings(settings)) if progress else None
    for ch, nm in (("L", "左"), ("R", "右")):
        r = ds.per[ch]
        (b_avg, b_pos), (a_avg, a_pos) = r["stats_lo"]
        (b2, _), (a2, _) = r["stats_hi"]
        lines.append(f"{nm}聲道：{fmin:.0f}–300 Hz 誤差 {b_avg:.1f} → {a_avg:.1f} dB（各位置 {b_pos:.1f} → {a_pos:.1f}）；"
                     f"300 Hz–{fmax / 1000:g} kHz {b2:.1f} → {a2:.1f} dB")
    lines.append(f"選定：修正 {fmin:.0f} Hz–{fmax / 1000:g} kHz，最大提升 {cand['max_boost']:g} dB，最大衰減 {cand['max_cut']:g} dB，"
                 f"平滑 1/{round(1 / cand['smooth'])} oct，位置間差異 {spread:.1f} dB"
                 + ("，啟用左右時間對齊" if align else "") + ("，高頻資料不可靠因此只修到 5 kHz" if not hf_ok else ""))
    return cand, lines, ds
