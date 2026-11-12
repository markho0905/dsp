"""roomcorr_pages.py - the six pages (量測 / 分析 / 目標曲線 / 濾波器 / 低頻控制 / 匯出) following the design deck."""
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)
from scipy.fft import rfftfreq

import roomcorr as R
import roomcorr_analysis as A
import roomcorr_dsp as D
from roomcorr_measure import KINDS, kinds_for, positions_for, tasks_for
from roomcorr_tabs import FAMILIES, Worker
from roomcorr_theme import (BRIGHT, COBALT, MUTED, ORANGE, TEXT, Card, DashedMeter, GradientButton, LegendChip,
                            MatrixView, MicView, RadioCard, Segmented, Toggle, ToggleBanner, h_title)
from roomcorr_widgets import CurveWidget, SpectrogramWidget, WaterfallWidget
import roomcorr_tf as T

TINY = 1e-12
KIND_NAMES = {"L": "左喇叭", "R": "右喇叭", "LR": "雙聲道"}


def kind_name(k):
    return KIND_NAMES.get(k, f"低音砲 {k[1:]}" if k.startswith("S") else k)


def pos_name(p):
    return {1: "中央", 2: "左 30 cm", 3: "右 30 cm"}.get(p, f"低音位置 {p}")


class LabeledSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, text, lo, hi, val, step=0.5, fmt="{:+.1f} dB"):
        super().__init__()
        self.step, self.fmt = step, fmt
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        row = QHBoxLayout()
        self.lb = QLabel(text)
        self.lb.setStyleSheet("font-weight:600;")
        self.val = QLabel("")
        self.val.setStyleSheet("font-weight:800;")
        row.addWidget(self.lb)
        row.addStretch(1)
        row.addWidget(self.val)
        v.addLayout(row)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setRange(int(round(lo / step)), int(round(hi / step)))
        self.sl.valueChanged.connect(self._ch)
        v.addWidget(self.sl)
        self.set(val)

    def _ch(self, i):
        self.val.setText(self.fmt.format(i * self.step))
        self.valueChanged.emit(i * self.step)

    def set(self, x):
        self.sl.blockSignals(True)
        self.sl.setValue(int(round(x / self.step)))
        self.sl.blockSignals(False)
        self.val.setText(self.fmt.format(self.sl.value() * self.step))

    def value(self):
        return self.sl.value() * self.step


def row_label(text):
    lb = QLabel(text)
    lb.setStyleSheet(f"color:{MUTED};font-size:12px;font-weight:600;")
    return lb


def clear_layout(lay):
    while lay.count():
        it = lay.takeAt(0)
        wd, sub = it.widget(), it.layout()
        if wd is not None:
            wd.hide()
            wd.setParent(None)
            wd.deleteLater()
        elif sub is not None:
            clear_layout(sub)


# ====================================================================================== 量測
class MeasurePage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.sel = ("L", 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        self.head = h_title("量測", "")
        top.addWidget(self.head, 1)
        self.badge = QLabel("")
        self.badge.setStyleSheet("background:#EAF0F9;border-radius:15px;padding:6px 14px;font-weight:700;")
        top.addWidget(self.badge)
        b = GradientButton("測量設定…", "outline")
        b.clicked.connect(win.open_measure_settings)
        top.addWidget(b)
        root.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(16)
        left = Card("麥克風位置", "請把麥克風放到標示的位置，高度與耳朵同高")
        self.mic = MicView()
        self.mic.micClicked.connect(self.on_mic)
        left.v.addWidget(self.mic, 1)
        row = QHBoxLayout()
        self.b_guided = GradientButton("依序量測全部", "outline")
        self.b_guided.clicked.connect(lambda: win.start_measure())
        self.b_level = GradientButton("測試音量", "outline")
        self.b_level.clicked.connect(win.do_level_check)
        row.addWidget(self.b_guided)
        row.addWidget(self.b_level)
        left.v.addLayout(row)
        self.lb_level = QLabel("")
        self.lb_level.setStyleSheet(f"color:{MUTED};")
        left.v.addWidget(self.lb_level)
        tip = QLabel("量測期間請保持安靜並關閉 Roon 播放。掃頻會以目前放大器音量播放，請先把音量調低。")
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color:{MUTED};font-size:12px;")
        left.v.addWidget(tip)
        body.addWidget(left, 5)

        right = QVBoxLayout()
        right.setSpacing(16)
        c = Card()
        hr = QHBoxLayout()
        t = QLabel("掃頻進度")
        t.setStyleSheet("font-size:16px;font-weight:700;")
        self.pct = QLabel("0%")
        self.pct.setStyleSheet("font-size:20px;font-weight:800;")
        hr.addWidget(t)
        hr.addStretch(1)
        hr.addWidget(self.pct)
        c.v.addLayout(hr)
        self.status = QLabel("按下開始，播放掃頻訊號")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        c.v.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        c.v.addWidget(self.bar)
        self.meter = DashedMeter()
        c.v.addWidget(self.meter)
        self.b_start = GradientButton("開始量測")
        self.b_start.setMinimumHeight(44)
        self.b_start.clicked.connect(self.on_start)
        c.v.addWidget(self.b_start)
        right.addWidget(c)

        c = Card("量測矩陣")
        self.matrix = MatrixView([], [])
        self.matrix.cellClicked.connect(self.on_cell)
        c.v.addWidget(self.matrix)
        right.addWidget(c)

        c = Card("麥克風")
        c.v.addWidget(row_label("輸入裝置"))
        self.cb_in = QComboBox()
        self.cb_in.activated.connect(lambda _: win.set_mcfg("m_in", self.cb_in.currentText()))
        c.v.addWidget(self.cb_in)
        c.v.addWidget(row_label("校準檔"))
        self.cal = GradientButton("（未使用）— 點此選擇", "outline")
        self.cal.clicked.connect(win.pick_cal)
        c.v.addWidget(self.cal)
        right.addWidget(c)
        right.addStretch(1)
        body.addLayout(right, 3)
        root.addLayout(body, 1)
        win.measureChanged.connect(self.refresh)
        self.refresh()

    # ---- helpers
    def kinds(self):
        return kinds_for(self.win.mcfg["nsub"])

    def positions(self):
        n = self.win.mcfg["npos"] if self.win.mcfg["nsub"] else 3
        return positions_for(max(3, n))

    def on_cell(self, kind, pos):
        if (kind, pos) in set(tasks_for(self.win.mcfg["nsub"], self.win.mcfg["npos"])):
            self.sel = (kind, pos)
            self.refresh()

    def on_mic(self, pos):
        if (self.sel[0], pos) in set(tasks_for(self.win.mcfg["nsub"], self.win.mcfg["npos"])):
            self.sel = (self.sel[0], pos)
            self.refresh()

    def on_start(self):
        if self.win.measuring:
            self.win.abort_measure()
        else:
            self.win.start_measure(only=self.sel)

    def refresh(self):
        w = self.win
        kinds, poss = self.kinds(), self.positions()
        valid = set(tasks_for(w.mcfg["nsub"], w.mcfg["npos"]))
        self.matrix.rebuild([(k, n) for k, n in kinds], [(p, pos_name(p)) for p, _ in poss])
        done = 0
        for (k, p), cell in self.matrix.cells.items():
            has = bool(w.session and w.session.has(k, p))
            cell.set_state(has, selected=(k, p) == self.sel, current=(k, p) == w.current_task, disabled=(k, p) not in valid)
        for k, p in valid:
            if w.session and w.session.has(k, p):
                done += 1
        self.badge.setText(f"已完成 {done} / {len(valid)}")
        self.head.sub_label.setText(f"位置：{pos_name(self.sel[1])} · {kind_name(self.sel[0])}")
        order = [2, 1, 3] + [p for p, _ in poss if p > 3]          # left, centre, right, then extra positions
        self.mic.set_mics([(p, pos_name(p)) for p in order if p in dict(poss)][:6])
        self.mic.set_active(self.sel[1], self.sel[0])
        self.b_start.setText("停止" if w.measuring else "開始量測")
        self.b_start.variant = "outline" if w.measuring else "primary"
        self.b_start.update()
        self.b_guided.setEnabled(not w.measuring)
        self.b_level.setEnabled(not w.measuring)
        self.cb_in.blockSignals(True)
        self.cb_in.clear()
        for name in w.mic_devices:
            self.cb_in.addItem(name)
        i = self.cb_in.findText(w.mcfg["m_in"])
        if i >= 0:
            self.cb_in.setCurrentIndex(i)
        self.cb_in.blockSignals(False)
        self.cal.setText(os.path.basename(w.mic_cal) if w.mic_cal else "（未使用）— 點此選擇")

    def set_progress(self, frac, text):
        self.bar.setValue(int(frac * 100))
        self.pct.setText(f"{int(frac * 100)}%")
        self.status.setText(text)
        self.meter.set_value(0.0 if frac <= 0 or frac >= 1 else 0.35 + 0.5 * abs(np.sin(frac * 40)))


# ====================================================================================== 分析
class AnalysisPage(QWidget):
    """Measurement analysis only: frequency response, spectrogram, waterfall (before / after correction) and the
    analysis report.  Filter settings live on 「濾波器」, the target on 「目標曲線」."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.sel = ("L", 1)
        self.view, self.state, self.span = "fr", "after", "full"
        self.cache = {}
        self.tf_worker = None
        self.tf_pending = None
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 8, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        self.head = h_title("分析", "")
        top.addWidget(self.head, 1)
        root.addLayout(top)

        c = Card()
        hr = QHBoxLayout()
        self.seg_view = Segmented([("頻率響應", "fr"), ("光譜圖", "spec"), ("瀑布圖", "wf")], "fr")
        self.seg_view.changed.connect(self.on_view)
        self.seg_state = Segmented([("校正前", "before"), ("校正後", "after")], "after", small=True)
        self.seg_state.changed.connect(self.on_state)
        self.seg_span = Segmented([("全頻", "full"), ("低頻 15–500 Hz", "bass")], "full", small=True)
        self.seg_span.changed.connect(self.on_span)
        hr.addWidget(self.seg_view)
        hr.addStretch(1)
        hr.addWidget(self.seg_state)
        hr.addWidget(self.seg_span)
        c.v.addLayout(hr)
        self.msg = QLabel("")
        self.msg.setStyleSheet(f"color:{MUTED};")
        c.v.addWidget(self.msg)
        self.charts = QStackedWidget()
        self.curve = CurveWidget()
        self.curve.editable = False
        self.curve.set_range(-20, 15, 5)
        self.curve.setMinimumHeight(300)
        self.spec = SpectrogramWidget()
        self.wf = WaterfallWidget()
        for wd in (self.curve, self.spec, self.wf):
            self.charts.addWidget(wd)
        c.v.addWidget(self.charts, 1)
        root.addWidget(c, 3)

        row = QHBoxLayout()
        row.setSpacing(16)
        c = Card("量測矩陣")
        self.matrix = MatrixView([], [])
        self.matrix.cellClicked.connect(self.on_cell)
        c.v.addWidget(self.matrix)
        c.v.addStretch(1)
        row.addWidget(c, 1)
        c = Card("分析報告")
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report.setStyleSheet(f"color:{TEXT};font-size:12px;line-height:150%;")
        c.v.addWidget(self.report)
        c.v.addStretch(1)
        row.addWidget(c, 2)
        root.addLayout(row, 0)
        win.analysisChanged.connect(self.refresh)
        win.designChanged.connect(self.refresh)
        win.measureChanged.connect(self.refresh_matrix)
        self.refresh()

    # ---- controls
    def on_view(self, v):
        self.view = v
        self.charts.setCurrentIndex({"fr": 0, "spec": 1, "wf": 2}[v])
        self.seg_span.setVisible(v != "fr")
        self.refresh_chart()

    def on_state(self, v):
        self.state = v
        self.refresh_chart()

    def on_span(self, v):
        self.span = v
        self.refresh_chart()

    def on_cell(self, kind, pos):
        self.sel = (kind, pos)
        self.refresh()

    def refresh_matrix(self):
        w = self.win
        kinds = [k for k in kinds_for(w.mcfg["nsub"]) if not k[0].startswith("S")]
        poss = positions_for(3)
        self.matrix.rebuild(kinds, [(p, pos_name(p)) for p, _ in poss])
        for (k, p), cell in self.matrix.cells.items():
            has = bool(w.session and w.session.has(k, p))
            cell.set_state(has, selected=(k, p) == self.sel, disabled=not has)

    def refresh(self):
        self.cache = {}
        self.refresh_matrix()
        self.head.sub_label.setText(f"{pos_name(self.sel[1])} · {kind_name(self.sel[0])}")
        w = self.win
        an, ds = w.analysis, w.design_result
        lines = list(an.report) if an is not None else []
        if an is not None and ds is not None:
            lines += an.design_text(ds)
        self.report.setText("\n".join("• " + ln for ln in lines) if lines else "完成量測與分析後，這裡會列出左右平衡、到達時間差、低頻峰谷、位置間差異與殘響時間。")
        self.seg_view.setVisible(True)
        self.seg_span.setVisible(self.view != "fr")
        self.refresh_chart()

    # ---- charts
    def refresh_chart(self):
        w = self.win
        an, ds = w.analysis, w.design_result
        if an is None or ds is None:
            self.curve.set_series([])
            self.spec.set_data(None, None, None)
            self.wf.set_data(None, None, None)
            self.msg.setText("尚未分析：請先到「量測」完成量測，或稍候計算中…" if an is None else "計算中…")
            return
        self.msg.setText("")
        if self.view == "fr":
            cv = an.curves_for(ds, *self.sel)
            if cv is None:
                self.curve.set_series([])
                self.msg.setText("這一格還沒有量測資料。")
                return
            ix = an.ix
            fx = an.f[ix]
            before, after = cv
            series = [("量測", ORANGE, fx, before[ix], 2.4, False),
                      ("目標", "#6B778F", fx, ds.tgt_rel[ix], 1.8, True),
                      ("校正後", BRIGHT, fx, after[ix], 2.6, False)]
            if self.sel[0] in ("L", "R"):
                series.append(("平均校正後", COBALT, fx, ds.per[self.sel[0]]["after"][ix], 2.2, False))
            self.curve.set_xrange(20, 20000)
            self.curve.set_series(series)
            return
        lo, hi = (15.0, 500.0) if self.span == "bass" else (20.0, 20000.0)
        wd = self.spec if self.view == "spec" else self.wf
        wd.set_xrange(lo, hi)
        key = (self.view, self.sel, self.state, self.span, id(ds))
        if key in self.cache:
            self._show(key, self.cache[key])
            return
        ir = self.ir_for(self.sel, self.state == "after")
        if ir is None:
            wd.set_data(None, None, None)
            self.msg.setText("這一格還沒有量測資料。")
            return
        self.msg.setText("計算中…")
        wd.empty_text = "計算中…"
        wd.set_data(None, None, None)
        self.tf_pending = key
        if self.tf_worker is not None:
            return
        self._start_tf(key, ir, an.fs)

    def _start_tf(self, key, ir, fs):
        view, span = key[0], key[3]
        if view == "spec":
            tmax = 0.8 if span == "bass" else 0.4
            fn = lambda prog: T.spectrogram(ir, fs, 15.0 if span == "bass" else 20.0, 500.0 if span == "bass" else 20000.0,
                                            tmax=tmax, cycles=8.0 if span == "bass" else 6.0)
        else:
            fn = lambda prog: T.waterfall(ir, fs, fmin=15.0 if span == "bass" else 20.0, fmax=500.0 if span == "bass" else 20000.0,
                                          tmax=0.4 if span == "bass" else 0.25, nslices=61 if span == "bass" else 41,
                                          win=0.3 if span == "bass" else 0.1, nf=420, smooth=1 / 24 if span == "bass" else 1 / 12)
        self.tf_worker = Worker(fn)
        self.tf_worker.done.connect(lambda r, k=key: self._tf_done(k, r))
        self.tf_worker.failed.connect(lambda m: (setattr(self, "tf_worker", None), self.msg.setText("計算失敗：" + m)))
        self.tf_worker.start()

    def _tf_done(self, key, res):
        self.tf_worker = None
        self.cache[key] = res
        if self.tf_pending == key:
            self._show(key, res)
        else:
            self.refresh_chart()

    def _show(self, key, res):
        t, f, Z = res
        self.msg.setText(("光譜圖：0 dB＝最強處；某個頻率往下拖得越長，代表餘音越久（駐波、殘響）。" if key[0] == "spec" else
                          "瀑布圖：最後方是 0 ms 的頻譜，往前是逐漸延遲後殘留的頻譜（頻率左低右高，紅＝最強）；往前仍高起的山脊就是駐波。右鍵拖曳可旋轉、滾輪縮放、雙擊重設。")
                         + ("（校正後＝量測 × 目前濾波器的預測）" if key[2] == "after" else ""))
        if key[0] == "spec":
            self.spec.set_data(t, f, Z, 60 if key[3] == "bass" else 50)
        else:
            self.wf.set_data(t, f, Z, -40 if key[3] == "bass" else -35)

    def ir_for(self, sel, after):
        """Impulse response of one matrix cell, optionally convolved with the designed filters (all paths, so a
        MIMO design shows the support speaker's contribution too)."""
        from scipy.signal import fftconvolve
        an, ds = self.win.analysis, self.win.design_result
        kind, pos = sel
        m = an.meas
        l, r = m.get("L", {}).get(pos), m.get("R", {}).get(pos)

        def add(*xs):
            xs = [x for x in xs if x is not None]
            n = max(len(x) for x in xs)
            return sum(np.pad(x, (0, n - len(x))) for x in xs)
        hx = getattr(ds, "hx", None) or {}
        if kind in ("L", "R"):
            ir = l if kind == "L" else r
            if ir is None:
                return None
            if not after:
                return np.asarray(ir, float)
            own = fftconvolve(ir, ds.h[kind])
            oth = r if kind == "L" else l
            x = hx.get(kind + ("R" if kind == "L" else "L"))
            return add(own, fftconvolve(oth, x)) if (x is not None and oth is not None) else own
        if after and l is not None and r is not None:
            a, b = fftconvolve(l, ds.h["L"]), fftconvolve(r, ds.h["R"])
            parts = [a, b]
            if "RL" in hx:
                parts.append(fftconvolve(l, hx["RL"]))
            if "LR" in hx:
                parts.append(fftconvolve(r, hx["LR"]))
            return add(*parts)
        if pos in an.both:
            return np.asarray(an.both[pos], float)
        if l is not None and r is not None:
            return add(l, r)
        return None


# ====================================================================================== 目標曲線
class TargetPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(h_title("目標曲線", "決定想聽到的平衡"), 1)
        root.addLayout(top)
        body = QHBoxLayout()
        body.setSpacing(16)
        c = Card("目標曲線", "拖曳藍點調整；雙擊空白處新增點；右鍵點刪除")
        self.curve = CurveWidget()
        self.curve.legend = False
        self.curve.set_range(-20, 15, 5)
        self.curve.released.connect(self.on_drag)
        self.curve.setMaximumHeight(560)
        c.v.addWidget(self.curve, 1)
        c.v.addStretch(1)
        body.addWidget(c, 3)
        c = Card("預設與自訂")
        self.radios = {}
        for name, sub in (("平坦", "全頻段回到同一水平"), ("低頻增強", "低頻緩緩抬升，聲音更厚實"),
                          ("高頻柔和", "高頻緩緩下降，更耐聽"), ("自訂", "調整低頻、高頻傾斜與上限頻率")):
            r = RadioCard(name, sub)
            r.clicked.connect(lambda _=False, n=name: self.pick(n))
            c.v.addWidget(r)
            self.radios[name] = r
        self.s_bass = LabeledSlider("低頻增強", -6, 10, 0, 0.5, "{:+.1f} dB")
        self.s_knee = LabeledSlider("低頻轉折", 40, 200, 100, 5, "{:.0f} Hz")
        self.s_tilt = LabeledSlider("高頻傾斜（20 kHz 處）", -8, 2, 0, 0.5, "{:+.1f} dB")
        self.s_fmax = LabeledSlider("校正上限頻率", 1000, 20000, 5000, 250, "{:.0f} Hz")
        for key, s in (("bass", self.s_bass), ("knee", self.s_knee), ("tilt", self.s_tilt)):
            s.valueChanged.connect(lambda v, k=key: self.custom(k, v))
            c.v.addWidget(s)
        self.s_fmax.valueChanged.connect(lambda v: win.update_setting("fmax", v))
        c.v.addWidget(self.s_fmax)
        note = QLabel("「校正上限頻率」以上的範圍不做校正。這裡選好的目標，分析、濾波器與低頻控制都會沿用。")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED};font-size:12px;")
        c.v.addWidget(note)
        c.v.addStretch(1)
        body.addWidget(c, 2)
        root.addLayout(body, 1)
        win.settingsChanged.connect(self.sync)
        win.stateChanged.connect(self.sync)
        win.designChanged.connect(self.plot_measured)
        self.sync()

    def pick(self, name):
        if name == "自訂":
            self.win.update_setting("preset", "自訂", clear_points=True)
        else:
            self.win.update_setting("preset", name, clear_points=True)

    def custom(self, key, v):
        s = self.win.cur
        self.win.update_setting(key, v, clear_points=True, preset="自訂")

    def on_drag(self):
        self.win.update_setting("points", [list(p) for p in self.curve.get_points()])

    def sync(self):
        w = self.win
        s = w.cur
        manual = bool(s.get("points"))
        for n, r in self.radios.items():
            r.setChecked(n == s["preset"] and not manual)
        self.s_bass.set(s["bass"])
        self.s_knee.set(s["knee"])
        self.s_tilt.set(s["tilt"])
        self.s_fmax.set(s["fmax"])
        for sl in (self.s_bass, self.s_knee, self.s_tilt):
            sl.setEnabled(s["preset"] == "自訂")
        self.curve.set_points(A.target_points_from(s))
        self.plot_measured()

    def plot_measured(self):
        an, ds = self.win.analysis, self.win.design_result
        if an is None or ds is None:
            self.curve.set_series([])
            return
        ix = an.ix
        fx = an.f[ix]
        self.curve.set_series([("L 量測", ORANGE, fx, ds.per["L"]["before"][ix], 1.6, False),
                               ("R 量測", "#8E6BD9", fx, ds.per["R"]["before"][ix], 1.6, False)])


# ====================================================================================== 濾波器
class FilterPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(h_title("濾波器", "調整濾波器參數；存檔請到「匯出」"), 1)
        root.addLayout(top)
        c = Card()
        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(14)
        self.phase = Segmented([("最小相位", "min"), ("線性相位", "linear"), ("混合", "mixed")], "mixed")
        self.taps = Segmented([("16384", 16384), ("32768", 32768), ("65536", 65536)], 16384)
        self.fs = Segmented([("44.1 kHz", 44100), ("48 kHz", 48000), ("96 kHz", 96000)], 48000)
        self.smooth = Segmented([("1/3 oct", 1 / 3), ("1/6 oct", 1 / 6), ("1/12 oct", 1 / 12)], 1 / 6)
        for i, (t, wd, key) in enumerate((("相位類型", self.phase, "phase"), ("濾波器長度（taps）", self.taps, "taps"),
                                          ("取樣率", self.fs, "out_fs"), ("平滑", self.smooth, "smooth"))):
            g.addWidget(row_label(t), (i // 2) * 2, i % 2)
            g.addWidget(wd, (i // 2) * 2 + 1, i % 2)
            wd.changed.connect(lambda v, k=key: win.update_setting(k, v))
        self.boost = LabeledSlider("最大提升", 0, 12, 6, 0.5, "{:.1f} dB")
        self.cut = LabeledSlider("最大衰減", 0, 24, 15, 0.5, "{:.1f} dB")
        self.boost.valueChanged.connect(lambda v: win.update_setting("max_boost", v))
        self.cut.valueChanged.connect(lambda v: win.update_setting("max_cut", v))
        g.addWidget(self.boost, 4, 0)
        g.addWidget(self.cut, 4, 1)
        c.v.addLayout(g)
        self.mimo = ToggleBanner("MIMO 聯合校正（支援喇叭）", "低頻由另一顆喇叭發出支援訊號，一起把每個量測點修平（Dirac ART 式）；頻率上限與模式在「進階設定」")
        self.mimo.toggled.connect(lambda v: win.update_setting("mimo", v))
        c.v.addWidget(self.mimo)
        self.keep_bass = ToggleBanner("保留低頻能量", "300 Hz 以下削峰與補谷互相抵銷，低音整體音量不變（關閉＝低頻對齊目標曲線）")
        self.keep_bass.toggled.connect(lambda v: win.update_setting("keep_bass", v))
        c.v.addWidget(self.keep_bass)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color:{MUTED};")
        c.v.addWidget(self.info)
        br = QHBoxLayout()
        b0 = GradientButton("進階設定…", "outline")
        b0.clicked.connect(win.open_advanced)
        bf = GradientButton("FIR 工具…", "outline")
        bf.clicked.connect(win.open_fir_tool)
        b1 = GradientButton("預覽（到分析頁）", "outline")
        b1.clicked.connect(lambda: (win.run_design(), win.goto("analysis")))
        ba = GradientButton("⚡ 自動最佳化並套用")
        ba.clicked.connect(win.auto_optimize)
        br.addWidget(b0, 1)
        br.addWidget(bf, 1)
        br.addWidget(b1, 1)
        br.addWidget(ba, 1)
        c.v.addLayout(br)
        root.addWidget(c, 0)
        c = Card("預覽：校正後頻率響應（平均）")
        self.curve = CurveWidget()
        self.curve.editable = False
        self.curve.set_range(-20, 15, 5)
        self.curve.setMaximumHeight(420)
        c.v.addWidget(self.curve, 1)
        root.addWidget(c, 1)
        win.settingsChanged.connect(self.sync)
        win.stateChanged.connect(self.sync)
        win.designChanged.connect(self.plot)
        self.sync()

    def sync(self):
        w = self.win
        s = w.cur
        self.phase.set_value(s["phase"])
        self.taps.set_value(int(s["taps"]))
        self.fs.set_value(int(s["out_fs"]))
        best = min((1 / 3, 1 / 6, 1 / 12), key=lambda x: abs(x - s["smooth"]))
        self.smooth.set_value(best)
        self.boost.set(s["max_boost"])
        self.cut.set(s["max_cut"])
        self.mimo.set_on(bool(s["mimo"]))
        self.keep_bass.set_on(bool(s.get("keep_bass", False)))
        fs = int(s["out_fs"])
        if s["phase"] == "linear":
            self.info.setText(f"線性相位的延遲 = 長度 ÷ 2 ÷ 取樣率；{int(s['taps'])} taps、{fs / 1000:g} kHz 時約 "
                              f"{s['taps'] / 2 / fs * 1000:.0f} ms。")
        elif s["phase"] == "min":
            self.info.setText("最小相位：沒有預振鈴，不修正相位（延遲接近 0）。")
        else:
            self.info.setText(f"混合：最小相位的幅度修正，加上 {s['phase_fmax']:.0f} Hz 以下的相位修正，延遲約 {s['delay_ms']:.0f} ms。")

    def plot(self):
        an, ds = self.win.analysis, self.win.design_result
        if an is None or ds is None:
            self.curve.set_series([])
            return
        ix, fx = an.ix, an.f[an.ix]
        self.curve.set_series([("L 校正後", BRIGHT, fx, ds.per["L"]["after"][ix], 2.2, False),
                               ("R 校正後", COBALT, fx, ds.per["R"]["after"][ix], 2.2, False),
                               ("目標", "#6B778F", fx, ds.tgt_rel[ix], 1.6, True)])


# ====================================================================================== 匯出
class ExportPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(h_title("匯出", "左右聲道各一個 WAV，同時存成目前模式的設定檔"), 1)
        root.addLayout(top)
        c = Card("輸出")
        c.v.addWidget(row_label("預設輸出資料夾"))
        row = QHBoxLayout()
        self.folder = QLineEdit(win.export_cfg["folder"])
        self.folder.editingFinished.connect(lambda: win.set_export("folder", self.folder.text()))
        b = GradientButton("選擇…", "outline")
        b.clicked.connect(self.choose)
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        c.v.addLayout(row)
        row = QHBoxLayout()
        self.ask = Toggle(win.export_cfg["ask"])
        self.ask.clicked.connect(lambda v: win.set_export("ask", v))
        t = QLabel("儲存前先詢問（遇到同名檔案，提醒會被覆蓋）")
        t.setStyleSheet("font-weight:600;")
        row.addWidget(self.ask)
        row.addWidget(t)
        row.addStretch(1)
        c.v.addLayout(row)
        c.v.addWidget(row_label("將寫出的檔案"))
        self.files = QLabel("")
        self.files.setStyleSheet(f"font-family:Menlo,monospace;font-size:13px;background:#F6F8FC;border-radius:10px;padding:12px;")
        self.files.setWordWrap(True)
        c.v.addWidget(self.files)
        self.note = QLabel("匯出時，同一組濾波器也會存成「%s」設定檔，可以從選單列一鍵切換校正與設定檔。")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{MUTED};font-size:12px;")
        c.v.addWidget(self.note)
        self.b = GradientButton("確認儲存")
        self.b.setMinimumHeight(46)
        self.b.clicked.connect(win.export_current)
        c.v.addWidget(self.b)
        root.addWidget(c, 0)

        c = Card("存檔前試聽")
        self.listen = ToggleBanner("▶ 試聽目前算好的濾波器", "存檔前先透過 DAC 聽這組結果；關掉即恢復原本的校正狀態")
        self.listen.toggled.connect(self.on_listen)
        c.v.addWidget(self.listen)
        self.lb_listen = QLabel("比較原始聲音：按右上角「Bypass」或鍵盤 B。")
        self.lb_listen.setStyleSheet(f"color:{MUTED};font-size:12px;")
        c.v.addWidget(self.lb_listen)
        root.addWidget(c, 0)
        root.addStretch(1)
        win.settingsChanged.connect(self.sync)
        win.stateChanged.connect(self.sync)
        win.designChanged.connect(self.sync)
        win.designChanged.connect(self.on_design)
        self.sync()

    # ---- audition of the designed (unsaved) filter
    def on_listen(self, on):
        if on and self.win.design_result is None:
            self.listen.set_on(False)
            self.lb_listen.setText("還沒有算好的濾波器。")
            return
        self.win.set_preview(on)
        self.lb_listen.setText("正在播放算好的濾波器（尚未存檔）；按 B 比較原始聲音" if on
                               else "比較原始聲音：按右上角「Bypass」或鍵盤 B。")

    def on_design(self):
        if self.listen.sw.isChecked():
            self.win.set_preview(True)

    def stop_listen(self):
        if self.listen.sw.isChecked():
            self.listen.set_on(False)
            self.on_listen(False)

    def choose(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", self.folder.text())
        if d:
            self.folder.setText(d)
            self.win.set_export("folder", d)
            self.sync()

    def sync(self):
        w = self.win
        names = w.export_file_names()
        folder = self.folder.text()
        lines = []
        for n in names:
            p = os.path.join(folder, n)
            lines.append(f"{n}" + ("　（已存在，會被覆蓋）" if os.path.exists(p) else ""))
        self.files.setText("\n".join(lines))
        self.note.setText(f"匯出時，同一組濾波器也會存成「{dict(A.SLOTS)[w.slot]}」設定檔，可以從選單列一鍵切換校正與設定檔。")
        self.ask.setChecked(bool(w.export_cfg["ask"]))
        self.b.setEnabled(w.design_result is not None)


# ====================================================================================== 低頻控制
class XoDialog(QDialog):
    """Crossover / bass-management / DAC channel settings for the multi-subwoofer system."""

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("分頻與輸出聲道")
        self.win = win
        c = win.bass_cfg
        f = QFormLayout(self)
        self.fc = QDoubleSpinBox()
        self.fc.setRange(40, 200)
        self.fc.setSuffix(" Hz")
        self.fc.setValue(c["fc"])
        self.fam = QComboBox()
        for n, k in FAMILIES:
            self.fam.addItem(n, k)
        self.fam.setCurrentIndex(max(self.fam.findData(c["fam"]), 0))
        self.order = QComboBox()
        for o in (2, 4, 6, 8):
            self.order.addItem(f"{o} 階", o)
        self.order.setCurrentIndex(max(self.order.findData(c["order"]), 0))
        self.phase = QComboBox()
        self.phase.addItem("線性相位", "linear")
        self.phase.addItem("自然相位", "natural")
        self.phase.setCurrentIndex(max(self.phase.findData(c["phase"]), 0))
        self.method = QComboBox()
        self.method.addItem("MSO 式：增益／延遲／極性／全通", "mso")
        self.method.addItem("FIR 最小平方（每顆低音砲一組 FIR）", "ls")
        self.method.setCurrentIndex(max(self.method.findData(c["method"]), 0))
        self.main = QComboBox()
        self.main.addItem("不修正主喇叭（只分頻）", "none")
        self.main.addItem("使用目前的主喇叭修正", "design")
        self.main.setCurrentIndex(max(self.main.findData(c["main"]), 0))
        self.eq = QCheckBox("最佳化後加整體 EQ")
        self.eq.setChecked(c["eq"])
        self.maxd = QDoubleSpinBox()
        self.maxd.setRange(0, 40)
        self.maxd.setSuffix(" ms")
        self.maxd.setValue(c["max_delay"])
        f.addRow("分頻頻率", self.fc)
        f.addRow("分頻類型", self.fam)
        f.addRow("階數", self.order)
        f.addRow("相位", self.phase)
        f.addRow("最佳化方法", self.method)
        f.addRow("最大延遲", self.maxd)
        f.addRow("主喇叭", self.main)
        f.addRow("", self.eq)
        self.ch = []
        for i, n in enumerate(("L", "R", "低音砲 1", "低音砲 2", "低音砲 3", "低音砲 4")):
            sp = QSpinBox()
            sp.setRange(1, 64)
            sp.setValue(c["channels"][i])
            self.ch.append(sp)
            f.addRow(f"DAC 輸出聲道：{n}", sp)
        b = GradientButton("完成")
        b.clicked.connect(self.accept)
        f.addRow(b)

    def accept(self):
        c = self.win.bass_cfg
        c.update(fc=self.fc.value(), fam=self.fam.currentData(), order=self.order.currentData(),
                 phase=self.phase.currentData(), method=self.method.currentData(), main=self.main.currentData(),
                 eq=self.eq.isChecked(), max_delay=self.maxd.value(), channels=[s.value() for s in self.ch])
        self.win.save_cfg()
        super().accept()


class BassPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.data = self.opt = None
        self.x = None
        self.x0 = None
        self.worker = None
        self.W = None
        self.applied = None
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(h_title("低頻控制", "設定低頻目標曲線，並控制最大提升與衰減"), 1)
        self.mode = Segmented([("MSO 多低音砲", "mso"), ("Bass Control", "bc")], "bc")
        self.mode.changed.connect(self.on_mode)
        top.addWidget(self.mode)
        root.addLayout(top)
        body = QHBoxLayout()
        body.setSpacing(16)
        c = Card()
        hr = QHBoxLayout()
        t = QLabel("低頻響應")
        t.setStyleSheet("font-size:16px;font-weight:700;")
        t2 = QLabel("20 – 200 Hz")
        t2.setStyleSheet(f"color:{MUTED};font-size:15px;font-weight:600;")
        hr.addWidget(t)
        hr.addWidget(t2)
        hr.addStretch(1)
        hr.addWidget(LegendChip("量測平均", ORANGE))
        hr.addWidget(LegendChip("目標", "#55617A", hollow=True))
        hr.addWidget(LegendChip("校正後", COBALT))
        c.v.addLayout(hr)
        self.curve = CurveWidget()
        self.curve.editable = False
        self.curve.legend = False
        self.curve.set_range(-25, 15, 5)
        self.curve.set_xrange(20, 200)
        self.curve.setMaximumHeight(560)
        c.v.addWidget(self.curve, 1)
        c.v.addStretch(1)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet(f"color:{MUTED};")
        c.v.addWidget(self.stats)
        body.addWidget(c, 3)

        self.stack = QStackedWidget()
        # --- Bass Control page
        bc = Card()
        note = QLabel("低頻目標（增益、轉折頻率）在「目標曲線」頁調整，最大提升／衰減在「濾波器」頁調整；這裡只放低頻專屬的選項。")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED};font-size:12px;")
        bc.v.addWidget(note)
        row = QHBoxLayout()
        for text, key in (("目標曲線…", "target"), ("濾波器…", "filter")):
            b = GradientButton(text, "outline")
            b.clicked.connect(lambda _=False, k=key: win.goto(k))
            row.addWidget(b, 1)
        bc.v.addLayout(row)
        self.t_decay = ToggleBanner("衰減控制", "縮短低頻餘音，讓低頻更俐落")
        self.t_limit = ToggleBanner("限制最大提升", "保護低音砲，避免過度推動")
        self.t_decay.toggled.connect(lambda v: win.update_setting("decay", v))
        self.t_limit.toggled.connect(lambda v: win.update_setting("limit_boost", v))
        bc.v.addWidget(self.t_decay)
        bc.v.addWidget(self.t_limit)
        bc.v.addStretch(1)
        self.stack.addWidget(bc)
        # --- MSO page
        self.mso = Card()
        self.mso_rows = QVBoxLayout()
        self.mso.v.addLayout(self.mso_rows)
        row = QHBoxLayout()
        self.b_auto = GradientButton("自動最佳化")
        self.b_auto.clicked.connect(self.auto)
        b_xo = GradientButton("分頻與輸出聲道…", "outline")
        b_xo.clicked.connect(self.open_xo)
        row.addWidget(self.b_auto, 1)
        row.addWidget(b_xo, 1)
        self.mso.v.addLayout(row)
        row = QHBoxLayout()
        b1 = GradientButton("重設", "outline")
        b1.clicked.connect(self.mso_reset)
        self.b_apply = GradientButton("套用到設定檔")
        self.b_apply.clicked.connect(self.mso_apply)
        row.addWidget(b1, 1)
        row.addWidget(self.b_apply, 1)
        self.mso.v.addLayout(row)
        self.mso_msg = QLabel("")
        self.mso_msg.setWordWrap(True)
        self.mso_msg.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.mso.v.addWidget(self.mso_msg)
        self.mso.v.addStretch(1)
        self.stack.addWidget(self.mso)
        body.addWidget(self.stack, 2)
        root.addLayout(body, 1)
        win.settingsChanged.connect(self.sync)
        win.stateChanged.connect(self.sync)
        win.designChanged.connect(self.plot)
        win.measureChanged.connect(self.reload_if_visible)
        self.sync()

    # ---- common
    def on_mode(self, v):
        self.stack.setCurrentIndex(0 if v == "bc" else 1)
        if v == "mso":
            self.reload()
        self.plot()

    def sync(self):
        s = self.win.cur
        self.t_decay.set_on(bool(s["decay"]))
        self.t_limit.set_on(bool(s["limit_boost"]))
        self.plot()

    def reload_if_visible(self):
        if self.mode.value() == "mso":
            self.reload()

    # ---- MSO (multi-subwoofer)
    def subs_available(self):
        s = self.win.session
        if s is None:
            return {}, None
        meas, _, fs = s.load()
        return {k: v for k, v in meas.items() if k.startswith("S") and v}, fs

    def reload(self):
        w = self.win
        subs, fs = self.subs_available()
        clear_layout(self.mso_rows)
        self.sl = []
        if not subs:
            self.data = self.opt = None
            self.mso_msg.setText("還沒有低音砲的量測。到「量測」→「測量設定…」把低音砲數量設為 1 以上，"
                                 "每顆低音砲在每個位置各量一次（建議 5 個位置以上）。")
            self.plot()
            return
        common = set.intersection(*[set(v) for v in subs.values()])
        if len(common) < 2:
            self.mso_msg.setText("至少需要 2 個位置，且每顆低音砲都要量測。")
            self.data = self.opt = None
            self.plot()
            return
        c = w.bass_cfg
        fmin, fmax = 20.0, 200.0
        nfft = 1 << 17
        self.fs = fs
        self.subs = subs
        self.data = D.BassData(subs, fs, fmin, fmax, nfft)
        f_full = rfftfreq(nfft, 1 / fs)
        self.lift = w.cur["bass"] / (1 + (f_full / 50.0) ** 2) if w.cur["preset"] == "自訂" else np.zeros_like(f_full)
        self.opt = D.BassOptimizer(self.data, fmin, fmax, target_db=(f_full, self.lift), max_delay_ms=c["max_delay"], n_allpass=2)
        self.x0 = self.opt.plain_x()
        if self.x is None or len(self.x) != len(self.x0):
            self.x = self.x0.copy()
        st = self.opt.stride
        for n in range(self.opt.S):
            name = self.data.names[n]
            self.mso_rows.addWidget(QLabel(f"低音砲 {name[1:]}" if name.startswith("S") else name))
            g = LabeledSlider("增益", -12, 3, float(self.x[n * st]), 0.5, "{:+.1f} dB")
            d = LabeledSlider("延遲", 0, c["max_delay"], float(self.x[n * st + 1]), 0.1, "{:.1f} ms")
            inv = QCheckBox("反相")
            inv.setChecked(bool(self.x[n * st + 2] > 0.5))
            if n == 0:
                d.setEnabled(False)
            g.valueChanged.connect(lambda v, n=n: self.set_x(n, 0, v))
            d.valueChanged.connect(lambda v, n=n: self.set_x(n, 1, v))
            inv.toggled.connect(lambda v, n=n: self.set_x(n, 2, 1.0 if v else 0.0))
            self.mso_rows.addWidget(g)
            self.mso_rows.addWidget(d)
            self.mso_rows.addWidget(inv)
            self.sl.append((g, d, inv))
        self.mso_msg.setText(f"已載入 {self.opt.S} 顆低音砲、{len(self.data.pos)} 個位置。可手動調整，或按「自動最佳化」。")
        self.plot()

    def set_x(self, n, k, v):
        self.x[n * self.opt.stride + k] = v
        self.plot()

    def push_to_sliders(self):
        st = self.opt.stride
        for n, (g, d, inv) in enumerate(self.sl):
            g.set(float(self.x[n * st]))
            d.set(float(self.x[n * st + 1]))
            inv.blockSignals(True)
            inv.setChecked(bool(self.x[n * st + 2] > 0.5))
            inv.blockSignals(False)

    def mso_reset(self):
        if self.opt is not None:
            self.x = self.x0.copy()
            self.W = None
            self.push_to_sliders()
            self.plot()

    def auto(self):
        if self.opt is None or self.worker is not None:
            return
        self.b_auto.setEnabled(False)
        self.mso_msg.setText("最佳化中…")
        opt = self.opt
        self.worker = Worker(lambda prog: (opt.run(iters=250, pop=18, progress=prog)))
        self.worker.progress.connect(self.mso_msg.setText)
        self.worker.done.connect(self.auto_done)
        self.worker.failed.connect(lambda m: (self.mso_msg.setText("失敗：" + m), self.b_auto.setEnabled(True), setattr(self, "worker", None)))
        self.worker.start()

    def auto_done(self, x):
        self.worker = None
        self.b_auto.setEnabled(True)
        self.x = np.array(x, float)
        self.push_to_sliders()
        s0, s1 = self.opt.stats(self.x0), self.opt.stats(self.x)
        self.mso_msg.setText(f"完成。RMS 偏差 {s0['rms']:.1f} → {s1['rms']:.1f} dB；位置間差異 {s0['spread']:.1f} → {s1['spread']:.1f} dB。")
        self.plot()

    def open_xo(self):
        XoDialog(self.win).exec()
        self.reload()

    def plot(self):
        w = self.win
        if self.mode.value() == "bc" or self.opt is None:
            an, ds = w.analysis, w.design_result
            if an is None or ds is None:
                self.curve.set_series([])
                self.stats.setText("完成量測與分析後，這裡會顯示低頻響應。" if self.mode.value() == "bc" else "")
                return
            f = an.f
            sel = (f >= 15) & (f <= 300)
            idx = np.where(sel)[0][::2]
            fx = f[idx]
            avg = lambda k: 10 * np.log10((10 ** (ds.per["L"][k][idx] / 10) + 10 ** (ds.per["R"][k][idx] / 10)) / 2)
            self.curve.set_xrange(20, 200)
            self.curve.set_series([("量測平均", ORANGE, fx, avg("before"), 2.4, False),
                                   ("目標", "#6B778F", fx, ds.tgt_rel[idx], 1.8, True),
                                   ("校正後", COBALT, fx, avg("after"), 2.6, False)])
            self.stats.setText("")
            return
        from scipy.ndimage import uniform_filter1d
        opt = self.opt
        Yb, Ya = opt.combined(self.x0), opt.combined(self.x)
        sm = lambda P: 10 * np.log10(uniform_filter1d(P, 13, axis=-1, mode="nearest") + TINY)     # ~1/6 octave
        pb, pa = np.abs(Yb) ** 2, np.abs(Ya) ** 2
        ref = float(np.median(sm(pb.mean(axis=0))))
        f = opt.f
        series = []
        for k in range(Yb.shape[0]):
            series.append(("", "#F5D3AE", f, sm(pb[k]) - ref, 1.0, False))
            series.append(("", "#B7D0F7", f, sm(pa[k]) - ref, 1.0, False))
        series.append(("量測平均", ORANGE, f, sm(pb.mean(axis=0)) - ref, 2.4, False))
        series.append(("目標", "#6B778F", f, opt.tgt, 1.8, True))
        series.append(("校正後", COBALT, f, sm(pa.mean(axis=0)) - ref, 2.6, False))
        self.curve.set_xrange(20, 200)
        self.curve.set_series(series)
        s0, s1 = opt.stats(self.x0), opt.stats(self.x)
        self.stats.setText(f"各位置合併：RMS 偏差 {s0['rms']:.1f} → {s1['rms']:.1f} dB；位置間差異 {s0['spread']:.1f} → {s1['spread']:.1f} dB")

    def mso_apply(self):
        """Compute crossover alignment and store the system so that 套用並匯出 includes the subwoofers."""
        if self.opt is None:
            QMessageBox.information(self, "沒有資料", "請先量測低音砲。")
            return
        w = self.win
        c = w.bass_cfg
        nfft = 1 << 17
        fs = self.fs
        f_full = rfftfreq(nfft, 1 / fs)
        eq = None
        if c["eq"]:
            Y = self.opt.combined(self.x)
            e = D.bass_eq(Y, self.opt.f, np.interp(self.opt.f, f_full, self.lift), 20.0, 200.0)
            eq = D.minphase(10 ** (np.interp(f_full, self.opt.f, e, left=0.0, right=0.0) / 20), nfft)
        G = self.opt.dense_G(self.x, f_full, eq)
        meas, _, _ = w.session.load()
        align, trim = dict(delay_ms=0.0, polarity=1, score=float("nan")), 0.0
        if 1 in self.data.pos and 1 in meas.get("L", {}) and 1 in meas.get("R", {}):
            Ym = self.data.full_spec(meas["L"][1]) + self.data.full_spec(meas["R"][1])
            Ys = sum(self.data.full_spec(self.subs[nm][1]) * g for nm, g in G)
            align = D.crossover_align(Ys, Ym, f_full, c["fc"], c["fam"], c["order"], c["phase"] == "linear", 15.0)
            band = (f_full >= 0.5 * c["fc"]) & (f_full <= 1.5 * c["fc"])
            trim = float(np.mean(20 * np.log10(np.abs(Ym[band]) + TINY)) - np.mean(20 * np.log10(np.abs(Ys[band]) + TINY)))
            msg = (f"分頻對齊：{'延遲低音砲' if align['delay_ms'] >= 0 else '延遲主喇叭'} {abs(align['delay_ms']):.2f} ms，"
                   f"{'反相' if align['polarity'] < 0 else '正相'}，音量微調 {trim:+.1f} dB。")
        else:
            msg = "⚠ 沒有中央位置的左右喇叭量測，無法自動對齊；已用 0 ms。"
        self.applied = dict(G=G, align=align, trim=trim, fs=fs, nfft=nfft)
        self.mso_msg.setText("已套用。" + msg + " 按「套用並匯出」會一起輸出主喇叭與低音砲。")
        w.status("低音管理已套用")
        w.stateChanged.emit()

    def build_outputs(self, main_h, taps):
        """Multi-output set (mains with high-pass + sub feeds) for the current slot, or None if not applied."""
        a = self.applied
        if not a:
            return None
        c = self.win.bass_cfg
        n = 2 + len(a["G"])
        outs, info = D.build_bass_system(a["fs"], taps, c["fc"], c["fam"], c["order"], c["phase"], a["G"], main_h,
                                         a["align"], a["trim"], c["channels"][:n], nfft=a["nfft"])
        return outs, info
