"""roomcorr_measure.py - measurement session storage and sweep playback."""
import glob
import json
import os
import time

import numpy as np

import roomcorr as R

POSITIONS = [(1, "中央（聆聽位置）"), (2, "左側 30 cm"), (3, "右側 30 cm")]
KINDS = [("L", "左喇叭"), ("R", "右喇叭"), ("LR", "雙聲道")]


def kinds_for(n_subs):
    return KINDS + [(f"S{i}", f"低音砲 {i}") for i in range(1, n_subs + 1)]


def positions_for(n):
    return POSITIONS + [(i, f"低音位置 {i}（聆聽區內自訂）") for i in range(4, n + 1)]


def tasks_for(n_subs, n_pos):
    """(kind, pos) in measuring order: mains at positions 1-3 (L, R, L+R), subs at positions 1..n_pos."""
    out = []
    for pos, _ in positions_for(max(n_pos, 3)):
        if pos <= 3:
            out += [(k, pos) for k, _ in KINDS]
        out += [(f"S{i}", pos) for i in range(1, n_subs + 1) if pos <= n_pos]
    return out


def run_sweep(fs, mic_dev, mic_ch, out_dev, out_chs, dur, level_db, repeats=1,
              f1=15.0, f2=20000.0, tail=1.5):
    """Play a log sweep on out_chs (1-based) and record the mic.
    Returns (full_ir, mic_peak_dBFS, snr_dB).  full_ir keeps absolute timing (constant device latency)."""
    import sounddevice as sd
    f2 = min(f2, 0.45 * fs)
    sweep = R.log_sweep(fs, dur, f1, f2) * 10 ** (level_db / 20)
    play = np.concatenate([np.zeros(int(0.3 * fs)), sweep, np.zeros(int(tail * fs))]).astype(np.float32)
    data = np.tile(play[:, None], (1, len(out_chs)))
    fulls, peaks = [], []
    for _ in range(repeats):
        rec = sd.playrec(data, samplerate=fs, input_mapping=[mic_ch], output_mapping=list(out_chs),
                         device=(mic_dev, out_dev), blocking=True)
        mic = rec[:, 0].astype(float)
        peaks.append(np.abs(mic).max())
        fulls.append(R.deconvolve(mic, sweep))
    p0 = int(np.argmax(np.abs(fulls[0][:len(fulls[0]) // 2])))
    acc = fulls[0].copy()
    for x in fulls[1:]:                      # align repeats on their main peak before averaging
        p = int(np.argmax(np.abs(x[:len(x) // 2])))
        acc += np.roll(x, p0 - p)
    full = acc / len(fulls)
    seg = full[p0 + int(1.1 * fs): p0 + int(1.45 * fs)]
    snr = 20 * np.log10(abs(full[p0]) / (seg.std() + 1e-15)) if len(seg) > 100 else float("nan")
    return full, 20 * np.log10(max(peaks) + 1e-12), float(snr)


def level_check(fs, mic_dev, mic_ch, out_dev, out_chs, level_db):
    """Short sweep; returns mic peak in dBFS."""
    import sounddevice as sd
    sweep = R.log_sweep(fs, 2.0, 40, 12000) * 10 ** (level_db / 20)
    play = np.concatenate([np.zeros(int(0.2 * fs)), sweep, np.zeros(int(0.3 * fs))]).astype(np.float32)
    data = np.tile(play[:, None], (1, len(out_chs)))
    rec = sd.playrec(data, samplerate=fs, input_mapping=[mic_ch], output_mapping=list(out_chs),
                     device=(mic_dev, out_dev), blocking=True)
    return 20 * np.log10(np.abs(rec).max() + 1e-12)


class Session:
    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "meas"), exist_ok=True)
        os.makedirs(os.path.join(root, "both"), exist_ok=True)
        self.meta_path = os.path.join(root, "session.json")
        if os.path.exists(self.meta_path):
            try:
                self.meta = json.load(open(self.meta_path))
            except Exception:  # noqa: BLE001
                self.meta = {}
            self.meta.setdefault("anchors", {})
            self.meta.setdefault("snr", {})
            self.meta.setdefault("fs", None)
            self.meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M"))
        else:
            self.meta = {"anchors": {}, "snr": {}, "fs": None, "created": time.strftime("%Y-%m-%d %H:%M")}
            self._save_meta()

    def _save_meta(self):
        json.dump(self.meta, open(self.meta_path, "w"), indent=1)

    @property
    def name(self):
        return os.path.basename(self.root)

    def path(self, kind, pos):
        return os.path.join(self.root, "both" if kind == "LR" else "meas", f"{kind}_{pos:02d}.npz")

    def has(self, kind, pos):
        return os.path.exists(self.path(kind, pos))

    def snr(self, kind, pos):
        return self.meta["snr"].get(f"{kind}{pos}")

    def save(self, kind, pos, full, fs, snr, ir_len=1.0):
        key = str(pos)
        if key not in self.meta["anchors"]:     # first measurement at this position sets the common time origin
            pk = int(np.argmax(np.abs(full[:len(full) // 2])))
            self.meta["anchors"][key] = pk - int(0.010 * fs)
        ir = R.crop(full, self.meta["anchors"][key], int(ir_len * fs))
        np.savez(self.path(kind, pos), ir=ir, fs=fs, ref=False)
        self.meta["snr"][f"{kind}{pos}"] = snr
        self.meta["fs"] = int(fs)
        self._save_meta()

    def delete(self, kind, pos):
        if self.has(kind, pos):
            os.remove(self.path(kind, pos))
        if not any(os.path.exists(os.path.join(self.root, d, f"{k}_{pos:02d}.npz"))
                   for d in ("meas", "both") for k in ("L", "R", "LR", "S1", "S2", "S3", "S4")):
            self.meta["anchors"].pop(str(pos), None)
            self._save_meta()

    def load(self):
        meas = {"L": {}, "R": {}}
        both = {}
        fs = self.meta.get("fs")
        for p in sorted(glob.glob(os.path.join(self.root, "meas", "*_*.npz"))):
            kind, pos = os.path.basename(p)[:-4].rsplit("_", 1)
            z = np.load(p)
            meas.setdefault(kind, {})[int(pos)] = z["ir"].astype(float)
            fs = int(z["fs"])
        for p in sorted(glob.glob(os.path.join(self.root, "both", "LR_*.npz"))):
            z = np.load(p)
            both[int(os.path.basename(p)[:-4].split("_")[1])] = z["ir"].astype(float)
        return meas, both, fs


def list_sessions(sess_dir):
    if not os.path.isdir(sess_dir):
        return []
    return sorted([d for d in os.listdir(sess_dir) if os.path.isdir(os.path.join(sess_dir, d))], reverse=True)
