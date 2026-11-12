"""roomcorr_tabs.py - the FIR tool: a chain of blocks (crossover / PEQ / all-pass / delay / gain / polarity) on top
of the current correction, drawn live against the measurement and the target, with a one-click optimiser."""
import json
import os
import shutil
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QMessageBox, QSpinBox, QTextEdit, QVBoxLayout, QWidget)
from scipy import signal
from scipy.io import wavfile

import roomcorr_dsp as D
import roomcorr_firopt as FO
from roomcorr_theme import COBALT, MUTED, ORANGE, Card, GradientButton, Segmented, Toggle
from roomcorr_widgets import CurveWidget

TINY = 1e-12
FAMILIES = [("Linkwitz-Riley", "linkwitz-riley"), ("Butterworth", "butterworth"), ("Bessel", "bessel")]
TYPES = [("參量 EQ", "peq"), ("全通（相位 EQ）", "allpass"), ("延遲", "delay"), ("增益", "gain"), ("極性反轉", "polarity"),
         ("分頻 低通", "lowpass"), ("分頻 高通", "highpass")]
SHAPES = [("峰谷 (peaking)", "peaking"), ("低頻擱架", "lowshelf"), ("高頻擱架", "highshelf"), ("陷波", "notch")]


class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    _alive = set()      # keeps running workers referenced: owners drop their reference in done/failed slots while
                        # run() is still unwinding, and a QThread deleted while running aborts the process

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        Worker._alive.add(self)
        self.finished.connect(self._release)

    def _release(self):
        Worker._alive.discard(self)

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(e))


def dspin(lo, hi, v, step=1.0, suf="", dec=2):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(v)
    s.setSingleStep(step)
    s.setSuffix(suf)
    return s


def describe(b):
    ph = {"linear": "線性", "natural": "自然"}.get(b.get("phase"), "")
    ch = {"L": "[左] ", "R": "[右] "}.get(b.get("ch", "LR"), "")
    auto = "⚡ " if b.get("auto") else ""
    t = b["type"]
    if t in ("lowpass", "highpass"):
        body = f"{'低通' if t == 'lowpass' else '高通'} {b['family']} {b['order']} 階 @ {b['fc']:g} Hz（{ph}相位）" + ("，反相" if b.get("invert") else "")
    elif t == "peq":
        body = f"EQ {dict(SHAPES)[b['shape']] if b['shape'] in dict(SHAPES) else b['shape']} {b['fc']:g} Hz  Q {b['q']:g}  {b['gain']:+g} dB（{ph}相位）"
    elif t == "allpass":
        body = f"全通 {b['fc']:g} Hz  Q {b['q']:g}"
    elif t == "delay":
        body = f"延遲 {b['ms']:g} ms"
    elif t == "gain":
        body = f"增益 {b['db']:+g} dB"
    else:
        body = "極性反轉"
    return auto + ch + body


# ====================================================================================== FIR tool
class FirToolTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.chain = []
        self.model = None
        self.worker = None
        self.metrics0 = None
        self._editing = None                  # index of the block being edited live
        self._loading = False
        self.live = True
        self.t_plot = QTimer(self)
        self.t_plot.setSingleShot(True)
        self.t_plot.timeout.connect(self.update_plot)
        self.t_aud = QTimer(self)
        self.t_aud.setSingleShot(True)
        self.t_aud.timeout.connect(self.push_audition)

        root = QHBoxLayout(self)
        root.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(12)
        root.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSpacing(12)
        root.addLayout(right, 2)

        # ---- chart card
        c = Card()
        hr = QHBoxLayout()
        self.seg_q = Segmented([("幅度", "mag"), ("相位", "phase"), ("群延遲", "gd")], "mag", small=True)
        self.seg_q.changed.connect(lambda *_: self.update_plot())
        self.seg_src = Segmented([("預測聽感", "pred"), ("只看這條鏈", "chain")], "pred", small=True)
        self.seg_src.changed.connect(lambda *_: self.update_plot())
        hr.addWidget(self.seg_q)
        hr.addSpacing(10)
        hr.addWidget(self.seg_src)
        hr.addStretch(1)
        hr.addWidget(QLabel("FIR 長度"))
        self.cb_taps = QComboBox()
        for t in (4096, 8192, 16384, 32768, 65536):
            self.cb_taps.addItem(f"{t} taps", t)
        self.cb_taps.setCurrentIndex(2)
        self.cb_taps.currentIndexChanged.connect(self.changed)
        hr.addWidget(self.cb_taps)
        c.v.addLayout(hr)
        self.lb_info = QLabel("")
        self.lb_info.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.lb_info.setWordWrap(True)
        c.v.addWidget(self.lb_info)
        self.plot = CurveWidget()
        self.plot.editable = False
        self.plot.set_range(-20, 15, 5)
        self.plot.setMinimumHeight(340)
        c.v.addWidget(self.plot, 1)
        left.addWidget(c, 3)

        # ---- block editor card
        c = Card("區塊", "選取清單中的區塊即可直接修改，圖表立即更新；「＋ 新增」用目前的數值加一個新區塊")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        self.cb_type = QComboBox()
        for n, k in TYPES:
            self.cb_type.addItem(n, k)
        self.cb_ch = QComboBox()
        for n, k in (("左＋右", "LR"), ("只有左", "L"), ("只有右", "R")):
            self.cb_ch.addItem(n, k)
        self.cb_family = QComboBox()
        for n, k in FAMILIES:
            self.cb_family.addItem(n, k)
        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 8)
        self.sp_order.setValue(4)
        self.sp_fc = dspin(10, 20000, 80, 1, " Hz", 1)
        self.sp_q = dspin(0.1, 20, 1.0, 0.1, "", 2)
        self.sp_gain = dspin(-24, 24, 0, 0.5, " dB", 1)
        self.cb_shape = QComboBox()
        for n, k in SHAPES:
            self.cb_shape.addItem(n, k)
        self.sp_ms = dspin(0, 500, 0, 0.05, " ms", 2)
        self.cb_phase = QComboBox()
        self.cb_phase.addItem("自然相位（最小相位，像類比／IIR）", "natural")
        self.cb_phase.addItem("線性相位（只改幅度，不改相位）", "linear")
        self.chk_inv = QCheckBox("反相")
        self.w = {"family": ("類型", self.cb_family), "order": ("階數", self.sp_order), "fc": ("頻率", self.sp_fc),
                  "q": ("Q", self.sp_q), "gain": ("增益", self.sp_gain), "shape": ("形狀", self.cb_shape),
                  "ms": ("延遲", self.sp_ms), "phase": ("相位", self.cb_phase)}
        g.addWidget(QLabel("種類"), 0, 0)
        g.addWidget(self.cb_type, 0, 1)
        g.addWidget(QLabel("聲道"), 0, 2)
        g.addWidget(self.cb_ch, 0, 3)
        g.addWidget(self.chk_inv, 0, 4, 1, 2)
        self.labels = {}
        for i, (k, (lab, wd)) in enumerate(self.w.items()):
            lb = QLabel(lab)
            self.labels[k] = lb
            g.addWidget(lb, 1 + i // 3, (i % 3) * 2)
            g.addWidget(wd, 1 + i // 3, (i % 3) * 2 + 1)
        self.b_add = GradientButton("＋ 新增", "outline")
        self.b_add.clicked.connect(self.add_block)
        g.addWidget(self.b_add, 0, 6)
        c.v.addLayout(g)
        self.cb_type.currentIndexChanged.connect(self.on_type)
        for wd in (self.cb_ch, self.cb_family, self.cb_shape, self.cb_phase):
            wd.currentIndexChanged.connect(self.edit_live)
        for wd in (self.sp_order, self.sp_fc, self.sp_q, self.sp_gain, self.sp_ms):
            wd.valueChanged.connect(self.edit_live)
        self.chk_inv.toggled.connect(self.edit_live)
        row = QHBoxLayout()
        self.lst = QListWidget()
        self.lst.setMinimumHeight(150)
        self.lst.currentRowChanged.connect(self.on_select)
        row.addWidget(self.lst, 1)
        col = QVBoxLayout()
        for text, fn in (("刪除選取", self.del_block), ("清空", self.clear), ("移除 ⚡ 自動區塊", self.clear_auto),
                         ("匯出 FIR（WAV）…", self.export_fir), ("套用到設定檔的 L／R…", self.apply_to_profile)):
            b = GradientButton(text, "outline")
            b.clicked.connect(fn)
            col.addWidget(b)
        col.addStretch(1)
        row.addLayout(col)
        c.v.addLayout(row)
        left.addWidget(c, 2)

        # ---- optimiser card
        c = Card("⚡ 一鍵最佳化", "用 IIR（低頻）＋線性相位 FIR（高頻）的 EQ、延遲與全通，讓預測聽感最接近目標曲線")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        self.o_boost = dspin(0, 12, 3, 0.5, " dB", 1)
        self.o_cut = dspin(0, 24, 12, 0.5, " dB", 1)
        self.o_n = QSpinBox()
        self.o_n.setRange(1, 24)
        self.o_n.setValue(10)
        self.o_fiir = dspin(100, 2000, 400, 50, " Hz", 0)
        self.o_fmin = dspin(0, 200, 0, 5, " Hz", 0)
        self.o_fmax = dspin(0, 20000, 0, 500, " Hz", 0)
        for i, (lab, wd) in enumerate((("最大提升", self.o_boost), ("最大衰減", self.o_cut), ("每聲道 EQ 數上限", self.o_n),
                                       ("IIR／線性相位分界", self.o_fiir), ("下限（0＝自動）", self.o_fmin), ("上限（0＝沿用濾波器頁）", self.o_fmax))):
            g.addWidget(QLabel(lab), i // 2, (i % 2) * 2)
            g.addWidget(wd, i // 2, (i % 2) * 2 + 1)
        c.v.addLayout(g)
        self.o_flags = {}
        for key, text in (("align", "左右相位對齊（延遲＋全通，提升低頻合成一致性）"), ("level", "左右音量對齊（只衰減較大聲的一邊）"),
                          ("speech", "語音清晰度：300 Hz–4 kHz 加權"), ("decay", "低頻衰減：餘音長的頻段加權（駐波優先）")):
            row = QHBoxLayout()
            t = Toggle(True)
            lb = QLabel(text)
            lb.setStyleSheet("font-size:12px;")
            row.addWidget(t)
            row.addWidget(lb, 1)
            c.v.addLayout(row)
            self.o_flags[key] = t
        self.b_opt = GradientButton("⚡ 一鍵最佳化")
        self.b_opt.clicked.connect(self.optimise)
        c.v.addWidget(self.b_opt)
        self.lb_opt = QLabel("")
        self.lb_opt.setWordWrap(True)
        self.lb_opt.setStyleSheet(f"color:{MUTED};font-size:12px;")
        c.v.addWidget(self.lb_opt)
        right.addWidget(c, 0)

        c = Card("評估（目前 → 加上這條鏈）", "每個指標都用量測資料預測；時間類指標用聽音位置的脈衝響應")
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet("font-family:Menlo,monospace;font-size:12px;")
        c.v.addWidget(self.txt, 1)
        right.addWidget(c, 1)

        self.build_model()
        self.on_type()
        self.refresh()

    # ---- model of the current correction
    def build_model(self):
        w = self.win
        an, ds = w.analysis, w.design_result
        if an is None or ds is None:
            self.model = None
            self.lb_info.setText("還沒有量測／濾波器：只能看這條鏈本身的響應。完成量測後再開啟，就能看到預測聽感與一鍵最佳化。")
            self.seg_src.set_value("chain")
            self.b_opt.setEnabled(False)
            return
        self.model = FO.Model(an, ds.h, ds.tgt_rel, self.fs(), fmax=float(w.cur["fmax"]))
        self.metrics0 = self.model.metrics([], self.taps())
        self.b_opt.setEnabled(True)

    def fs(self):
        return self.win.audio_rate()

    def taps(self):
        return self.cb_taps.currentData()

    # ---- block editor
    def on_type(self, *_):
        t = self.cb_type.currentData()
        show = {"lowpass": ("family", "order", "fc", "phase"), "highpass": ("family", "order", "fc", "phase"),
                "peq": ("shape", "fc", "q", "gain", "phase"), "allpass": ("fc", "q"), "delay": ("ms",), "gain": ("gain",),
                "polarity": ()}[t]
        for k, (_, wd) in self.w.items():
            self.labels[k].setVisible(k in show)
            wd.setVisible(k in show)
        self.chk_inv.setVisible(t in ("lowpass", "highpass"))
        if t == "allpass" and self.sp_q.value() > 5:
            self.sp_q.setValue(0.7)
        if not self._loading and self._editing is not None:
            self.edit_live()

    def block_from_editor(self):
        t = self.cb_type.currentData()
        ph = self.cb_phase.currentData()
        ch = self.cb_ch.currentData()
        if t in ("lowpass", "highpass"):
            fam, order = self.cb_family.currentData(), self.sp_order.value()
            if fam == "linkwitz-riley" and order % 2:
                order += 1
            b = dict(type=t, family=fam, order=order, fc=self.sp_fc.value(), phase=ph, invert=self.chk_inv.isChecked())
        elif t == "peq":
            b = dict(type="peq", shape=self.cb_shape.currentData(), fc=self.sp_fc.value(), q=self.sp_q.value(),
                     gain=self.sp_gain.value(), phase=ph)
        elif t == "allpass":
            b = dict(type="allpass", fc=self.sp_fc.value(), q=self.sp_q.value())
        elif t == "delay":
            b = dict(type="delay", ms=self.sp_ms.value())
        elif t == "gain":
            b = dict(type="gain", db=self.sp_gain.value())
        else:
            b = dict(type="polarity")
        if ch != "LR":
            b["ch"] = ch
        return b

    def editor_from_block(self, b):
        self._loading = True
        self.cb_type.setCurrentIndex(max(self.cb_type.findData(b["type"]), 0))
        self.cb_ch.setCurrentIndex(max(self.cb_ch.findData(b.get("ch", "LR")), 0))
        if "family" in b:
            self.cb_family.setCurrentIndex(max(self.cb_family.findData(b["family"]), 0))
        if "order" in b:
            self.sp_order.setValue(int(b["order"]))
        if "fc" in b:
            self.sp_fc.setValue(float(b["fc"]))
        if "q" in b:
            self.sp_q.setValue(float(b["q"]))
        if "gain" in b:
            self.sp_gain.setValue(float(b["gain"]))
        if "db" in b:
            self.sp_gain.setValue(float(b["db"]))
        if "shape" in b:
            self.cb_shape.setCurrentIndex(max(self.cb_shape.findData(b["shape"]), 0))
        if "ms" in b:
            self.sp_ms.setValue(float(b["ms"]))
        if "phase" in b:
            self.cb_phase.setCurrentIndex(max(self.cb_phase.findData(b["phase"]), 0))
        self.chk_inv.setChecked(bool(b.get("invert")))
        self._loading = False
        self.on_type()

    def on_select(self, i):
        if i < 0 or i >= len(self.chain):
            self._editing = None
            return
        self._editing = i
        self.editor_from_block(self.chain[i])

    def edit_live(self, *_):
        """Any change in the editor rewrites the selected block and redraws at once."""
        if self._loading or self._editing is None or self._editing >= len(self.chain):
            return
        b = self.block_from_editor()
        self.chain[self._editing] = b
        self.lst.item(self._editing).setText(describe(b))
        self.changed()

    def add_block(self):
        self.chain.append(self.block_from_editor())
        self.refresh(select=len(self.chain) - 1)

    def del_block(self):
        i = self.lst.currentRow()
        if 0 <= i < len(self.chain):
            del self.chain[i]
            self.refresh(select=min(i, len(self.chain) - 1))

    def clear(self):
        self.chain = []
        self.refresh()

    def clear_auto(self):
        self.chain = [b for b in self.chain if not b.get("auto")]
        self.refresh()

    def refresh(self, select=None):
        self._editing = None
        self.lst.blockSignals(True)
        self.lst.clear()
        for b in self.chain:
            self.lst.addItem(describe(b))
        self.lst.blockSignals(False)
        if select is not None and 0 <= select < len(self.chain):
            self.lst.setCurrentRow(select)
        self.changed()

    def changed(self, *_):
        self.t_plot.start(40)
        if self.live:
            self.t_aud.start(400)

    # ---- live audition (A/B with the global Bypass / B key)
    def chain_firs(self):
        out = {}
        for c in "LR":
            blocks = FO.chain_for(self.chain, c)
            out[c] = D.chain_fir(blocks, self.fs(), self.taps())[0] if blocks else None
        return out

    def push_audition(self):
        if not self.chain:
            self.win.set_audition(None)
            return
        h = self.chain_firs()
        self.win.set_audition({c: (v if v is not None else np.array([1.0])) for c, v in h.items()})

    def stop_audition(self):
        self.live = False
        self.t_aud.stop()
        self.win.set_audition(None, bypass=False)

    # ---- chart
    def update_plot(self, *_):
        q, src = self.seg_q.value(), self.seg_src.value()
        m = self.model
        yunit = {"mag": "dB", "phase": "°", "gd": "ms"}[q]
        self.plot.yunit = yunit
        series = []
        if m is not None and src == "pred":
            fg = m.fg
            if q == "mag":
                series = [("目標", "#6B778F", fg, m.tgt, 1.6, True),
                          ("左 目前", "#F5C89A", fg, m.base["L"], 1.4, False), ("右 目前", "#C9B6EC", fg, m.base["R"], 1.4, False),
                          ("左 加上鏈", ORANGE, fg, m.mag("L", self.chain), 2.4, False),
                          ("右 加上鏈", "#8E6BD9", fg, m.mag("R", self.chain), 2.4, False)]
                self.plot.set_range(-20, 15, 5)
            elif m.has_center:
                for c, col0, col1, name in (("L", "#F5C89A", ORANGE, "左"), ("R", "#C9B6EC", "#8E6BD9", "右")):
                    H0, H1 = m.center_resp(c, []), m.center_resp(c, self.chain)
                    if q == "phase":
                        series += [(f"{name} 目前", col0, fg, np.degrees(np.angle(H0)), 1.2, False),
                                   (f"{name} 加上鏈", col1, fg, np.degrees(np.angle(H1)), 2.0, False)]
                    else:
                        series += [(f"{name} 目前", col0, fg, m.gd_ms(H0), 1.2, False), (f"{name} 加上鏈", col1, fg, m.gd_ms(H1), 2.0, False)]
                if q == "phase":
                    self.plot.set_range(-180, 180, 60)
                else:
                    ys = np.concatenate([s[3][(fg >= 30) & (fg <= 10000)] for s in series])
                    lo, hi = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
                    pad = max((hi - lo) * 0.15, 2.0)
                    self.plot.set_range(np.floor(lo - pad), np.ceil(hi + pad), max(1, int(round((hi - lo + 2 * pad) / 8))))
            else:
                self.lb_info.setText("沒有中央位置的左右量測，無法顯示相位／群延遲的預測。")
        if not series:                                            # chain only
            fs, nfft = self.fs(), 1 << 16
            f, H, lin = D.chain_response(self.chain, fs, nfft)
            ix = np.unique(np.geomspace(1, len(f) - 1, 900).astype(int))
            for c, col, name in (("L", COBALT, "左"), ("R", "#8E6BD9", "右")):
                Hc = np.ones(len(f), complex)
                for b in FO.chain_for(self.chain, c):
                    Hc = Hc * D.block_response(b, f, fs, nfft)
                if q == "mag":
                    y = 20 * np.log10(np.abs(Hc) + TINY)
                elif q == "phase":
                    y = np.degrees(np.angle(Hc))
                else:
                    y = -np.gradient(np.unwrap(np.angle(Hc)), f) / (2 * np.pi) * 1000
                series.append((f"{name} 鏈", col, f[ix], y[ix], 2.0, False))
            if q == "mag":
                self.plot.set_range(-30, 15, 5)
            elif q == "phase":
                self.plot.set_range(-180, 180, 60)
            else:
                ys = np.concatenate([s[3][(s[2] > 20) & (s[2] < 20000)] for s in series])
                hi = max(5.0, float(np.percentile(ys, 99)) * 1.2)
                lo = min(0.0, float(np.percentile(ys, 1)) * 1.2)
                self.plot.set_range(lo, hi, max(1, int(round((hi - lo) / 8))))
        self.plot.set_xrange(20, 20000)
        self.plot.set_series(series)
        taps, fs = self.taps(), self.fs()
        lin = any(b.get("phase") == "linear" for b in self.chain)
        lat = taps // 2 / fs * 1000 if lin else 0.0
        n_auto = sum(1 for b in self.chain if b.get("auto"))
        self.lb_info.setText(f"{len(self.chain)} 個區塊（{n_auto} 個自動）；{fs} Hz；"
                             + (f"含線性相位區塊，額外延遲約 {lat:.0f} ms" if lin else "無額外延遲（全部自然相位）")
                             + "。這條鏈正即時套用在播放上；按 B 比較原始聲音。")
        if m is not None:
            self.update_metrics()

    def update_metrics(self):
        m = self.model
        if m is None or self.metrics0 is None:
            return
        try:
            cur = m.metrics(self.chain, self.taps())
        except Exception as e:  # noqa: BLE001
            self.txt.setPlainText("評估失敗：" + str(e))
            return
        self.txt.setPlainText("\n".join(FO.metrics_table(self.metrics0, cur)))

    # ---- one-click optimiser
    def optimise(self):
        if self.model is None or self.worker is not None:
            return
        o = FO.Options(max_boost=self.o_boost.value(), max_cut=self.o_cut.value(), n_peq=self.o_n.value(),
                       f_iir=self.o_fiir.value(), fmin=self.o_fmin.value() or None, fmax=self.o_fmax.value() or None,
                       **{k: t.isChecked() for k, t in self.o_flags.items()})
        base = [b for b in self.chain if not b.get("auto")]
        m = self.model
        self.b_opt.setEnabled(False)
        self.lb_opt.setText("最佳化中…")
        self.worker = Worker(lambda prog: FO.optimise(m, base, o, prog))
        self.worker.progress.connect(self.lb_opt.setText)
        self.worker.done.connect(self._opt_done)
        self.worker.failed.connect(lambda msg: (setattr(self, "worker", None), self.b_opt.setEnabled(True), self.lb_opt.setText("失敗：" + msg)))
        self.worker.start()

    def _opt_done(self, res):
        self.worker = None
        self.b_opt.setEnabled(True)
        chain, notes = res
        self.chain = chain
        self.lb_opt.setText("\n".join(notes) + "\n已加入 ⚡ 區塊並即時套用；可逐一選取微調，或「移除 ⚡ 自動區塊」還原。")
        self.refresh()

    # ---- export / apply
    def export_fir(self):
        if not self.chain:
            QMessageBox.information(self, "空的", "請先加入至少一個區塊。")
            return
        p, _ = QFileDialog.getSaveFileName(self, "匯出 FIR（會寫出 _L / _R 兩個檔）", "fir.wav", "WAV (*.wav)")
        if not p:
            return
        base, ext = os.path.splitext(p)
        h = self.chain_firs()
        for c in "LR":
            x = h[c] if h[c] is not None else np.array([1.0], float)
            wavfile.write(f"{base}_{c}{ext or '.wav'}", self.fs(), x.astype(np.float32))
        QMessageBox.information(self, "完成", f"已寫出 {base}_L{ext or '.wav'} 與 {base}_R{ext or '.wav'}。")

    def apply_to_profile(self):
        """Convolve the per-channel chain into the current slot's L / R filters (old files kept as .bak)."""
        w = self.win
        if not self.chain:
            QMessageBox.information(self, "空的", "請先加入至少一個區塊。")
            return
        folder = w.slot_folder()
        if not os.path.isdir(folder):
            QMessageBox.information(self, "沒有設定檔", "目前的設定檔還沒有匯出。請先到「匯出」按「確認儲存」。")
            return
        if QMessageBox.question(self, "套用到設定檔", f"把這條 FIR 鏈卷積進「{os.path.basename(folder)}」的 L／R 濾波器？"
                                "（原檔會備份成 .bak）") != QMessageBox.Yes:
            return
        h = self.chain_firs()
        files = {}
        meta = os.path.join(folder, "profile.json")
        if os.path.exists(meta):
            for o in json.load(open(meta))["outputs"]:
                if o["name"] in ("L", "R"):
                    for p in o["paths"]:
                        files[p["file"]] = o["name"]
        else:
            from roomcorr_engine import find_filter_files
            lp, rp = find_filter_files(folder)
            files[os.path.basename(lp)] = "L"
            files[os.path.basename(rp)] = "R"
        for fn, ch in files.items():
            if h[ch] is None:
                continue
            path = os.path.join(folder, fn)
            shutil.copy(path, path + ".bak")
            fs, x = wavfile.read(path)
            if x.dtype.kind == "i":
                x = x / float(np.iinfo(x.dtype).max)
            y = signal.fftconvolve(x.astype(float), h[ch])
            wavfile.write(path, fs, y.astype(np.float32))
        self.stop_audition()
        self.live = True
        self.chain = []
        self.refresh()
        w.reload_profiles_and_apply()
        self.build_model()
        self.update_plot()
        QMessageBox.information(self, "完成", "已套用到設定檔並重新載入；這條鏈已清空（現在聽到的就是套用後的結果）。")
