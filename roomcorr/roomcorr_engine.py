"""
roomcorr_engine.py - real-time FIR convolution matrix for RoomCorr.

Stereo in (e.g. BlackHole 2ch fed by Roon) -> any number of output channels, each a sum of FIR-filtered inputs
(mains L/R, plus optional bass-managed subwoofer feeds).  One-click bypass / profile switching with a short
crossfade.  No GUI code in here.
"""
import json
import os
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile


# ------------------------------------------------------------------ filters
def read_wav_mono(path):
    fs, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x / float(np.iinfo(x.dtype).max)
    elif x.dtype.kind == "u":
        x = (x.astype(float) - 128) / 128
    x = x.astype(np.float64)
    return fs, (x if x.ndim == 1 else x[:, 0])


def find_filter_files(folder):
    files = {f.lower(): os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")}
    left = files.get("l.wav") or files.get("left.wav")
    right = files.get("r.wav") or files.get("right.wav")
    if left or right:
        return left or right, right or left
    if len(files) == 1:
        p = next(iter(files.values()))
        return p, p
    raise ValueError(f"{folder}: expected L.wav and R.wav (or exactly one wav file)")


class Profile:
    """A set of outputs; each output = list of (input_index, FIR, gain).  Legacy form: Profile(name, left, right, fs)."""

    def __init__(self, name, left, right, fs, outputs=None):
        self.name, self.fs = name, fs
        if outputs is None:
            outputs = [dict(name="L", channel=0, paths=[(0, left, 1.0)]),
                       dict(name="R", channel=1, paths=[(1, right, 1.0)])]
        self.outputs = outputs
        self.n_out = max(o["channel"] for o in outputs) + 1
        n = 1 << 17
        f = rfftfreq(n, 1 / fs)
        band = (f >= 300) & (f <= 3000)
        lv, pk = [], []
        for o in outputs:
            if o["channel"] > 1:
                continue
            F = sum(np.abs(rfft(h * g, n)) for _, h, g in o["paths"])
            lv.append(np.sqrt(np.mean(F[band] ** 2)))
            pk.append(F.max())
        self.mid_gain = float(np.clip(np.mean(lv), 1e-3, 10.0)) if lv else 1.0     # linear 300 Hz-3 kHz gain of the filter
        self.match = float(np.clip(1.0 / self.mid_gain, 0.25, 4.0))                  # legacy (no longer applied)
        self.peak_gain = max(pk) if pk else 1.0
        self.has_subs = any(o["channel"] > 1 for o in outputs)

    @classmethod
    def from_folder(cls, folder, stream_fs):
        def load(p):
            fs, h = read_wav_mono(p)
            if fs != stream_fs:
                fr = Fraction(stream_fs, fs).limit_denominator(1000)
                h = signal.resample_poly(h, fr.numerator, fr.denominator)
            return h
        name = os.path.basename(folder.rstrip("/"))
        meta = os.path.join(folder, "profile.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            outs, cache = [], {}
            for o in m["outputs"]:
                paths = []
                for p in o["paths"]:
                    if p["file"] not in cache:
                        cache[p["file"]] = load(os.path.join(folder, p["file"]))
                    paths.append((int(p["input"]), cache[p["file"]], float(p.get("gain", 1.0))))
                outs.append(dict(name=o["name"], channel=int(o["channel"]) - 1, paths=paths))
            return cls(name, None, None, stream_fs, outputs=outs)
        lp, rp = find_filter_files(folder)
        return cls(name, load(lp), load(rp), stream_fs)


# ------------------------------------------------------------------ convolution
class MatrixConv:
    """Uniform partitioned overlap-save convolution of n_in inputs into n_out outputs (shared input FFTs)."""

    def __init__(self, outputs, B, n_in=2):
        self.B, self.n_in = B, n_in
        self.n_out = max(o["channel"] for o in outputs) + 1
        self.paths = []
        for o in outputs:
            for inp, h, g in o["paths"]:
                P = int(np.ceil(len(h) / B))
                hp = np.zeros(P * B)
                hp[:len(h)] = np.asarray(h) * g
                self.paths.append((o["channel"], inp, rfft(hp.reshape(P, B), 2 * B, axis=1), P))
        self.Pmax = max(p[3] for p in self.paths)
        self.fdl = np.zeros((n_in, self.Pmax, B + 1), complex)
        self.prev = np.zeros((n_in, B))
        self.idx = 0
        self.ar = np.arange(self.Pmax)

    def process(self, x):
        """x: (B, n_in) -> (B, n_out)."""
        xt = x.T
        self.fdl[:, self.idx] = rfft(np.concatenate((self.prev, xt), axis=1), axis=1)
        self.prev = xt.copy()
        Y = np.zeros((self.n_out, self.B + 1), complex)
        for out, inp, H, P in self.paths:
            order = (self.idx - self.ar[:P]) % self.Pmax
            Y[out] += (H * self.fdl[inp][order]).sum(axis=0)
        self.idx = (self.idx + 1) % self.Pmax
        return irfft(Y, axis=1)[:, self.B:].T


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self):
        self.stream = None
        self.fs = 48000
        self.B = 1024
        self.n_out = 2
        self.level_match = True
        self.gain_db = 0.0
        self.peak = np.zeros(2)
        self.clipped = False
        self.xruns = 0
        self._convs = {}
        self._ref = None            # profile the bypass path is level-matched against
        self._want = None
        self._cur = None
        self._from = False
        self._fade_pos = 0
        self._g_prev = 1.0
        self.fade_len = int(0.08 * self.fs)

    # ---- control (GUI thread)
    def configure(self, fs, block, n_out=2):
        self.fs, self.B, self.n_out = fs, block, max(2, n_out)
        self.fade_len = max(int(0.08 * fs), block)
        self._convs.clear()
        self.peak = np.zeros(self.n_out)

    def prepare(self, prof):
        if prof is not None and id(prof) not in self._convs:
            self._convs[id(prof)] = MatrixConv(prof.outputs, self.B)

    def forget(self, prof):
        if prof is not None and prof not in (self._cur, self._from, self._want):
            self._convs.pop(id(prof), None)

    def select(self, prof, ref=None):
        """prof=None -> bypass (mains pass straight through, attenuated to the mid-band level of `ref` /
        the last correction when level_match is on, so A/B compares balance rather than loudness)."""
        self.prepare(prof)
        if prof is not None:
            self._ref = prof
        elif ref is not None:
            self._ref = ref
        self._want = prof

    def bypass_gain(self):
        return float(self._ref.mid_gain) if (self.level_match and self._ref is not None) else 1.0

    def bypass_gain_db(self):
        return 20 * np.log10(max(self.bypass_gain(), 1e-6))

    @property
    def active(self):
        return self._cur

    # ---- audio thread
    def _render(self, slot, x):
        out = np.zeros((len(x), self.n_out))
        if slot is None:
            out[:, :2] = x * self.bypass_gain()
            return out
        mc = self._convs.get(id(slot))
        if mc is None:
            out[:, :2] = x * self.bypass_gain()
            return out
        y = mc.process(x)
        out[:, :min(y.shape[1], self.n_out)] = y[:, :self.n_out]
        return out

    def process_block(self, x):
        want = self._want
        if want is not self._cur:
            self._from, self._cur, self._fade_pos = self._cur, want, 0
        y = self._render(self._cur, x)
        if self._from is not False:
            yo = self._render(self._from, x)
            w = np.clip((self._fade_pos + np.arange(len(x))) / self.fade_len, 0, 1)[:, None]
            y = yo * (1 - w) + y * w
            self._fade_pos += len(x)
            if self._fade_pos >= self.fade_len:
                self._from = False
        g = 10 ** (self.gain_db / 20)
        y = y * np.linspace(self._g_prev, g, len(x))[:, None]
        self._g_prev = g
        pk = np.abs(y).max(axis=0)
        self.peak = np.maximum(pk, self.peak * 0.9)
        if pk.max() > 1.0:
            self.clipped = True
        return np.clip(y, -1.0, 1.0)

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            self.xruns += 1
        if frames != self.B:
            outdata[:] = 0
            outdata[:, :min(2, outdata.shape[1])] = indata[:, :min(2, outdata.shape[1])]
            return
        outdata[:] = self.process_block(indata.astype(np.float64)).astype(np.float32)

    def start(self, in_dev, out_dev):
        import sounddevice as sd
        self.stop()
        self.stream = sd.Stream(samplerate=self.fs, blocksize=self.B, device=(in_dev, out_dev),
                                channels=(2, self.n_out), dtype="float32", latency="high",
                                callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

    @property
    def running(self):
        return self.stream is not None and self.stream.active
