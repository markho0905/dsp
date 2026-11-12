#!/usr/bin/env python3
"""mimo_gui_check.py - end-to-end check of the MIMO path through the real app, off-screen (no audio hardware).

Creates an isolated HOME, writes a synthetic measurement session (mimo_selftest.build_session's room, saved the
way the app saves sweeps), starts MainWindow with audio disabled, runs analysis, switches MIMO on, exports, and
then feeds the exported profile through the real-time engine to prove the support (cross) paths are convolved.
Screenshots of the filter / analysis pages are written to --shots (optional).

    QT_QPA_PLATFORM=offscreen python3 mimo_gui_check.py --shots /tmp/shots
"""
import argparse
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="roomcorr-home-")
os.environ["HOME"] = _home
os.environ["XDG_CONFIG_HOME"] = os.path.join(_home, ".config")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                     # noqa: E402
from scipy.io import wavfile                           # noqa: E402

import mimo_selftest as S                              # noqa: E402
import roomcorr_dsp as D                               # noqa: E402
from roomcorr_measure import Session                   # noqa: E402
from roomcorr_paths import SESS_DIR                    # noqa: E402


def wait_for(app, cond, timeout=120.0, what=""):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timeout waiting for {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None)
    a = ap.parse_args()
    fs = 48000
    # ---- synthetic session, saved like the app does after each sweep
    spk = {"L": (1.3, 1.0, 1.0), "R": (3.9, 1.0, 1.0)}
    mics = {1: (2.6, 3.0, 1.15), 2: (2.3, 3.0, 1.15), 3: (2.9, 3.0, 1.15)}
    sess = Session(os.path.join(SESS_DIR, "synthetic"))
    for pos, mp in mics.items():
        for i, c in enumerate("LR"):
            full = S.room_ir(fs, spk[c], mp, seed=10 * pos + i)
            full = np.concatenate([np.zeros(int(0.1 * fs)), full, np.zeros(int(0.3 * fs))])   # device latency
            sess.save(c, pos, full, fs, 60.0)
    # ---- app without audio (sounddevice is stubbed when PortAudio is not available on this machine)
    try:
        import sounddevice  # noqa: F401
    except Exception:  # noqa: BLE001
        import types
        sd = types.ModuleType("sounddevice")
        sd.query_devices = lambda *x, **k: []
        sd.query_hostapis = lambda *x, **k: []
        sd.default = types.SimpleNamespace(device=(None, None), samplerate=fs)
        sd._terminate = sd._initialize = lambda: None
        sd.stop = lambda: None
        sys.modules["sounddevice"] = sd
    from PySide6.QtWidgets import QApplication
    import roomcorr_app as RA
    RA.MainWindow.start_audio = lambda self, *x, **k: None
    RA.MainWindow.save_cfg = lambda self: None
    app = QApplication.instance() or QApplication(sys.argv)
    w = RA.MainWindow()
    w.resize(1280, 860)
    w.show()
    w.session = sess
    w.run_analysis()
    wait_for(app, lambda: w.analysis is not None, what="analysis")
    wait_for(app, lambda: w.design_result is not None and w.dworker is None, what="SISO design")
    siso = w.design_result
    assert not siso.mimo_used and not siso.hx
    # ---- switch MIMO on from the filter page toggle (as the user would)
    w.goto("filter")
    w.pages["filter"].mimo.set_on(True)
    w.update_setting("mimo", True)
    wait_for(app, lambda: w.design_result is not None and w.dworker is None and w.design_result.mimo_used, what="MIMO design")
    ds = w.design_result
    assert set(ds.hx) == {"LR", "RL"}, ds.hx.keys()
    an = w.analysis
    print("design text:")
    for line in an.design_text(ds):
        print("  ", line)
    # analysis-page predictions include the support paths
    for kind in ("L", "R", "LR"):
        b, af = an.curves_for(ds, kind, 1)
        ir = w.pages["analysis"].ir_for((kind, 1), True)
        assert np.all(np.isfinite(af)) and ir is not None and np.abs(ir).max() > 0
    # FIR tool model with cross filters
    import roomcorr_firopt as FO
    model = FO.Model(an, ds.h, ds.tgt_rel, fs, fmax=5000.0, hx=ds.hx)
    met = model.metrics([], 16384)
    print(f"FIR-tool model: rms_lo_L {met['rms_lo_L']:.1f} dB, decay median {met['decay_med']:.0f} ms, C50 {met['c50']:.1f} dB")
    # advanced dialog builds with the new MIMO controls
    dlg = RA.AdvancedDialog(w)
    assert dlg.mimo_mode.count() == 2
    dlg.deleteLater()
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
        for page in ("filter", "analysis"):
            w.goto(page)
            for _ in range(20):
                app.processEvents()
                time.sleep(0.02)
            w.grab().save(os.path.join(a.shots, f"{page}.png"))
        print("screenshots in", a.shots)
    # ---- export: 4 wavs + a profile.json with cross paths in the slot folder
    out = tempfile.mkdtemp(prefix="roomcorr-export-")
    w.export_cfg.update(folder=out, ask=False)
    w.export_current(quiet=True)
    for _ in range(10):
        app.processEvents()
    names = sorted(os.listdir(out))
    print("exported:", names)
    assert names == ["std_L.wav", "std_L_fromR.wav", "std_R.wav", "std_R_fromL.wav"], names
    slot = w.slot_folder()
    assert os.path.exists(os.path.join(slot, "profile.json")), os.listdir(slot)
    # ---- the real-time engine runs the 2x2 matrix
    from roomcorr_engine import Engine, Profile
    prof = Profile.from_folder(slot, fs)
    assert all(len(o["paths"]) == 2 for o in prof.outputs), [len(o["paths"]) for o in prof.outputs]
    eng = Engine()
    eng.configure(fs, 1024, 2)
    eng.select(prof)
    rng = np.random.default_rng(0)
    n = 47 * 1024  # ~1 s, whole blocks
    x = np.zeros((n, 2))
    x[:, 0] = rng.normal(size=n) * 0.1                                     # left programme only
    y = np.concatenate([eng.process_block(x[i:i + 1024]) for i in range(0, n, 1024)])
    f = np.fft.rfftfreq(n, 1 / fs)
    Y = np.abs(np.fft.rfft(y, axis=0))
    X = np.abs(np.fft.rfft(x[:, 0]))
    lo, hi = (f >= 40) & (f <= 120), (f >= 400) & (f <= 2000)
    r_lo = 20 * np.log10(Y[lo, 1].mean() / X[lo].mean() + 1e-12)
    r_hi = 20 * np.log10(Y[hi, 1].mean() / X[hi].mean() + 1e-12)
    print(f"engine: left-only input -> right speaker output: {r_lo:.1f} dB at 40-120 Hz, {r_hi:.1f} dB at 400-2000 Hz")
    assert r_lo > r_hi + 20, "support path should act only at low frequencies"
    assert np.abs(y[:, 0]).max() > 0
    # ---- bass-management assembly keeps the cross paths
    main_h = {c: ds.h[c] / ds.scale for c in "LR"}
    main_h.update({k: v / ds.scale for k, v in ds.hx.items()})
    G = [("S1", np.ones(an.nfft // 2 + 1, complex))]
    outs, _ = D.build_bass_system(fs, 16384, 80.0, "LR", 4, "min", G, main_h, dict(delay_ms=0.0, polarity=1), 0.0,
                                  [1, 2, 3], nfft=an.nfft)
    assert len(outs[0]["paths"]) == 2 and len(outs[0]["paths"][1]) == 4
    bass_dir = tempfile.mkdtemp(prefix="roomcorr-bass-")
    D.save_multi_profile(bass_dir, fs, outs)
    pb = Profile.from_folder(bass_dir, fs)
    assert pb.n_out == 3 and len(pb.outputs[0]["paths"]) == 2
    print("bass-managed profile with support paths:", sorted(os.listdir(bass_dir)))
    print("\nOK: MIMO design, display, export and real-time engine all work with the support paths.")
    w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
