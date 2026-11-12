#!/usr/bin/env python3
"""RoomCorr - 房間修正（量測 → 分析 → 目標曲線 → 濾波器 → 低頻控制 → 匯出）with one-click correction switch."""
import json
import os
import sys
import time
import traceback
from math import gcd

import numpy as np
from PySide6.QtCore import QEvent, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QStackedWidget, QSystemTrayIcon, QVBoxLayout,
                               QWidget)
from scipy import signal
from scipy.io import wavfile

import roomcorr as R
import roomcorr_analysis as A
import roomcorr_dsp as D
from roomcorr_engine import Engine, Profile
from roomcorr_measure import Session, kinds_for, level_check, list_sessions, run_sweep, tasks_for, positions_for
from roomcorr_pages import (AnalysisPage, BassPage, ExportPage, FilterPage, MeasurePage, TargetPage, kind_name,
                            pos_name)
from roomcorr_paths import APP_DIR, LOG_FILE, PROFILE_DIR, SESS_DIR
from roomcorr_tabs import FirToolTab, Worker
from roomcorr_theme import (BG, COBALT, MUTED, STYLE, CapsuleToggle, DeviceCard, GradientButton, NavButton, Segmented, nav_icon,
                            paint_gradient_rect)

RATES = [44100, 48000, 88200, 96000, 176400, 192000]
BLOCKS = [256, 512, 1024, 2048, 4096]
PAGES = [("measure", "mic", "量測"), ("analysis", "chart", "分析"), ("target", "wave", "目標曲線"),
         ("filter", "sliders", "濾波器"), ("bass", "pulse", "低頻控制"), ("export", "download", "匯出")]
DEFAULT_FOLDER = os.path.expanduser("~/Desktop/AutoEq_Project/filters")


def dot_icon(color):
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(10, 10, 44, 44)
    p.end()
    return QIcon(pm)


def logo_pixmap(size=34):
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    from PySide6.QtCore import QRectF
    paint_gradient_rect(p, QRectF(0, 0, size * 2, size * 2), size * 0.5, glow=False)
    ic = nav_icon("logo", "white", size)
    p.drawPixmap(int(size * 0.5), int(size * 0.5), ic)
    p.end()
    pm.setDevicePixelRatio(2)
    return pm


# ====================================================================================== dialogs
class AudioDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("輸出裝置與播放設定")
        self.win = win
        a = win.acfg
        f = QFormLayout(self)
        self.cb_in, self.cb_out = QComboBox(), QComboBox()
        for i, n in win.play_in:
            self.cb_in.addItem(n, i)
        for i, n in win.play_out:
            self.cb_out.addItem(n, i)
        self.cb_in.setCurrentIndex(max(self.cb_in.findText(a["in"]), 0))
        self.cb_out.setCurrentIndex(max(self.cb_out.findText(a["out"]), 0))
        self.cb_rate, self.cb_block = QComboBox(), QComboBox()
        for r in RATES:
            self.cb_rate.addItem(f"{r} Hz", r)
        for b in BLOCKS:
            self.cb_block.addItem(f"{b} samples", b)
        self.cb_rate.setCurrentIndex(max(self.cb_rate.findData(a["rate"]), 0))
        self.cb_block.setCurrentIndex(max(self.cb_block.findData(a["block"]), 0))
        self.match = QCheckBox("Bypass 時把原始聲音衰減到濾波器的中頻音量（公平 A/B，不會削波）")
        self.match.setChecked(a["match"])
        self.trim = QSlider(Qt.Horizontal)
        self.trim.setRange(-200, 0)
        self.trim.setValue(int(a["trim"]))
        self.lb_trim = QLabel(f"{a['trim'] / 10:.1f} dB")
        self.trim.valueChanged.connect(lambda v: self.lb_trim.setText(f"{v / 10:.1f} dB"))
        row = QHBoxLayout()
        row.addWidget(self.trim, 1)
        row.addWidget(self.lb_trim)
        f.addRow("輸入（BlackHole）", self.cb_in)
        f.addRow("輸出（DAC）", self.cb_out)
        self.b_rescan = QPushButton("找不到裝置？重新掃描")
        self.b_rescan.clicked.connect(self.rescan)
        f.addRow("", self.b_rescan)
        self.devlist = QLabel(win.device_report())
        self.devlist.setWordWrap(True)
        self.devlist.setStyleSheet("color:#55617A;font-size:11px;")
        f.addRow("系統看到的裝置", self.devlist)
        f.addRow("取樣率", self.cb_rate)
        f.addRow("緩衝", self.cb_block)
        f.addRow("", self.match)
        f.addRow("輸出微調", row)
        self.meters = []
        for n in ("L", "R"):
            pb = QProgressBar()
            pb.setRange(0, 600)
            pb.setTextVisible(False)
            self.meters.append(pb)
            f.addRow(n, pb)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#C2570A;")
        f.addRow(self.warn)
        self.b = GradientButton("")
        self.b.clicked.connect(self.toggle)
        f.addRow(self.b)
        self.tm = QTimer(self)
        self.tm.timeout.connect(self.tick)
        self.tm.start(60)
        self.refresh()

    def rescan(self):
        w = self.win
        was = w.engine.running
        if was:
            w.engine.stop()
        w.scan_devices(rescan=True)
        cur_in, cur_out = self.cb_in.currentText(), self.cb_out.currentText()
        self.cb_in.clear()
        self.cb_out.clear()
        for i, n in w.play_in:
            self.cb_in.addItem(n, i)
        for i, n in w.play_out:
            self.cb_out.addItem(n, i)
        self.cb_in.setCurrentIndex(max(self.cb_in.findText(cur_in), 0))
        self.cb_out.setCurrentIndex(max(self.cb_out.findText(cur_out), 0))
        self.devlist.setText(w.device_report())
        if was:
            self.apply()
            w.start_audio()
        self.refresh()

    def apply(self):
        a = self.win.acfg
        a.update({"in": self.cb_in.currentText(), "out": self.cb_out.currentText(), "rate": self.cb_rate.currentData(),
                  "block": self.cb_block.currentData(), "match": self.match.isChecked(), "trim": self.trim.value()})
        self.win.engine.level_match = a["match"]
        self.win.engine.gain_db = a["trim"] / 10
        self.win.save_cfg()
        self.win.update_device_card()

    def toggle(self):
        self.apply()
        if self.win.engine.running:
            self.win.engine.stop()
        else:
            self.win.start_audio()
        self.win.update_device_card()
        self.refresh()

    def refresh(self):
        self.b.setText("停止音訊" if self.win.engine.running else "開始音訊")

    def tick(self):
        e = self.win.engine
        for pb, p in zip(self.meters, e.peak):
            pb.setValue(int(np.clip((20 * np.log10(max(p, 1e-6)) + 60) * 10, 0, 600)))
        pr = self.win.profiles.get(self.win.slot)
        msg = ""
        if pr is not None:
            peak = 20 * np.log10(pr.peak_gain)
            if peak > 0.5:
                msg = f"此濾波器最大提升 +{peak:.1f} dB，大音量時可能削波：把「輸出微調」調到約 -{peak:.0f} dB 以下。"
        self.warn.setText(msg)

    def done(self, r):
        self.apply()
        super().done(r)


class MeasureDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("測量設定")
        self.win = win
        m = win.mcfg
        f = QFormLayout(self)

        def spin(lo, hi, v, suf=""):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(int(v))
            s.setSuffix(suf)
            return s
        self.out = QComboBox()
        for n in win.out_devices:
            self.out.addItem(n)
        self.out.setCurrentIndex(max(self.out.findText(m["m_out"]), 0))
        self.micch, self.lout, self.rout = spin(1, 16, m["mic_ch"]), spin(1, 64, m["lout"]), spin(1, 64, m["rout"])
        self.rate = QComboBox()
        for r in RATES:
            self.rate.addItem(f"{r} Hz", r)
        self.rate.setCurrentIndex(max(self.rate.findData(m["rate"]), 0))
        self.dur = QComboBox()
        for d in (6, 8, 12):
            self.dur.addItem(f"{d} 秒", d)
        self.dur.setCurrentIndex(max(self.dur.findData(m["dur"]), 0))
        self.level = spin(-60, -3, m["level"], " dBFS")
        self.rep = spin(1, 4, m["rep"])
        self.nsub = spin(0, 4, m["nsub"])
        self.npos = spin(3, 9, m["npos"])
        self.subch = [spin(1, 64, v) for v in m["subch"]]
        f.addRow("輸出裝置（DAC）", self.out)
        b_rs = QPushButton("找不到裝置？重新掃描")
        b_rs.clicked.connect(self.rescan)
        f.addRow("", b_rs)
        f.addRow("麥克風聲道", self.micch)
        f.addRow("左喇叭輸出聲道", self.lout)
        f.addRow("右喇叭輸出聲道", self.rout)
        f.addRow("取樣率", self.rate)
        f.addRow("掃頻長度", self.dur)
        f.addRow("掃頻音量", self.level)
        f.addRow("每項重複次數", self.rep)
        f.addRow("低音砲數量（0＝不量）", self.nsub)
        f.addRow("低音砲測量位置數", self.npos)
        for i, s in enumerate(self.subch):
            f.addRow(f"低音砲 {i + 1} 輸出聲道", s)
        b = GradientButton("完成")
        b.clicked.connect(self.accept)
        f.addRow(b)

    def rescan(self):
        w = self.win
        cur = self.out.currentText()
        if not w.engine.running:
            w.scan_devices(rescan=True)
        else:
            w.scan_devices()
        self.out.clear()
        for n in w.out_devices:
            self.out.addItem(n)
        self.out.setCurrentIndex(max(self.out.findText(cur), 0))
        w.measureChanged.emit()

    def accept(self):
        m = self.win.mcfg
        m.update(m_out=self.out.currentText(), mic_ch=self.micch.value(), lout=self.lout.value(), rout=self.rout.value(),
                 rate=self.rate.currentData(), dur=self.dur.currentData(), level=self.level.value(),
                 rep=self.rep.value(), nsub=self.nsub.value(), npos=self.npos.value(),
                 subch=[s.value() for s in self.subch])
        self.win.save_cfg()
        self.win.measureChanged.emit()
        super().accept()


class AdvancedDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("進階設定")
        self.win = win
        s = win.cur
        f = QFormLayout(self)

        def dsp(lo, hi, v, suf="", step=1.0):
            d = QDoubleSpinBox()
            d.setRange(lo, hi)
            d.setValue(float(v))
            d.setSuffix(suf)
            d.setSingleStep(step)
            return d
        self.delay = dsp(5, 100, s["delay_ms"], " ms")
        self.pfmax = dsp(0, 20000, s["phase_fmax"], " Hz", 100)
        self.dip = dsp(0, 30, s["dip"], " dB")
        self.pre = dsp(1, 100, s["pre_ms"], " ms")
        self.head = dsp(0, 12, s["headroom"], " dB", 0.5)
        self.align = QCheckBox("左右時間對齊")
        self.align.setChecked(bool(s["align"]))
        f.addRow("混合相位的濾波器延遲", self.delay)
        f.addRow("相位修正上限（0＝關；可到 20 kHz）", self.pfmax)
        f.addRow("避免填深谷（0＝關）", self.dip)
        f.addRow("預振鈴上限（需小於延遲）", self.pre)
        f.addRow("預留餘量", self.head)
        f.addRow("", self.align)
        ok = GradientButton("完成")
        ok.clicked.connect(self.accept)
        f.addRow(ok)

    def accept(self):
        self.win.update_many(dict(delay_ms=self.delay.value(), phase_fmax=self.pfmax.value(), dip=self.dip.value(),
                                  pre_ms=self.pre.value(), headroom=self.head.value(), align=self.align.isChecked()))
        super().accept()


# ====================================================================================== main window
class MainWindow(QMainWindow):
    stateChanged = Signal()
    measureChanged = Signal()
    analysisChanged = Signal()
    designChanged = Signal()
    settingsChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("roomcorr")
        self.cfg = QSettings("RoomCorr", "RoomCorr")
        self.engine = Engine()
        self.audition_h, self.audition_bypass = None, False
        self.preview_prof, self.preview_bypass = None, False
        self.global_bypass = False
        self.profiles = {}
        self.enabled = False
        self.slot = "std"
        self.session = None
        self.analysis = None
        self.design_result = None
        self.dworker = self.aworker = self.mworker = None
        self.d_dirty = False
        self.measuring = False
        self.current_task = None
        self.m_aborted = False
        self.engine_was_running = False
        self.mqueue = []
        os.makedirs(PROFILE_DIR, exist_ok=True)
        os.makedirs(SESS_DIR, exist_ok=True)

        self.load_cfg()

        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        main = QWidget()
        mv = QVBoxLayout(main)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        mv.addWidget(self._build_topbar())
        self.stack = QStackedWidget()
        self.pages = {}
        for key, cls in (("measure", MeasurePage), ("analysis", AnalysisPage), ("target", TargetPage),
                         ("filter", FilterPage), ("bass", BassPage), ("export", ExportPage)):
            self.pages[key] = cls(self)
            self.stack.addWidget(self.pages[key])
        mv.addWidget(self.stack, 1)
        h.addWidget(main, 1)
        # one-key bypass: B anywhere in the app, including dialogs and while a spin box has focus
        # (an application event filter, because a focused QLineEdit/spin box would otherwise swallow the key)
        QApplication.instance().installEventFilter(self)
        self.stateChanged.connect(self.sync_topbar)

        # tray
        self.tray = QSystemTrayIcon(dot_icon("#9AA5BB"), self)
        self.tray_menu = QMenu()
        self.a_toggle = QAction("校正 開／關", self)
        self.a_toggle.triggered.connect(lambda: self.set_enabled(not self.enabled))
        self.tray_menu.addAction(self.a_toggle)
        self.tray_menu.addSeparator()
        self.slot_actions = {}
        for sid, name in A.SLOTS:
            act = QAction(name, self)
            act.setCheckable(True)
            act.triggered.connect(lambda _=False, s=sid: self.set_slot(s))
            self.tray_menu.addAction(act)
            self.slot_actions[sid] = act
        self.tray_menu.addSeparator()
        for text, fn in (("顯示視窗", self.show_raise), ("結束", QApplication.quit)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            self.tray_menu.addAction(a)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(lambda r: self.set_enabled(not self.enabled) if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()

        self.design_timer = QTimer(self)
        self.design_timer.setSingleShot(True)
        self.design_timer.timeout.connect(self.run_design)
        self.reload_profiles()
        last = self.cfg.value("session", "")
        if last and os.path.isdir(os.path.join(SESS_DIR, last)):
            try:
                self.session = Session(os.path.join(SESS_DIR, last))
            except Exception as e:  # noqa: BLE001
                print("ignoring unreadable session", last, e)
        self.engine.level_match = self.acfg["match"]
        self.engine.gain_db = self.acfg["trim"] / 10
        self.goto("measure")
        self.refresh_all()
        self.statusBar().setStyleSheet(f"color:{MUTED};background:{BG};")
        if self.session is not None and self.session.has("L", 1):
            QTimer.singleShot(400, self.run_analysis)
        if self.cfg.value("autostart", True, bool):
            QTimer.singleShot(500, self.start_audio)

    # ------------------------------------------------------------ global top bar (mode, correction, bypass)
    def _build_topbar(self):
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(28, 14, 28, 0)
        lay.setSpacing(12)
        lay.addStretch(1)
        self.top_slots = Segmented([(n, k) for k, n in A.SLOTS], self.slot)
        self.top_slots.changed.connect(self.set_slot)
        self.top_enable = CapsuleToggle()
        self.top_enable.toggled.connect(self.set_enabled)
        self.top_bypass = GradientButton("Bypass（B）", "outline")
        self.top_bypass.setMinimumWidth(170)
        self.top_bypass.setToolTip("一鍵比較原始聲音；任何頁面按鍵盤 B 也可以切換")
        self.top_bypass.clicked.connect(self.toggle_bypass)
        lay.addWidget(self.top_slots)
        lay.addWidget(self.top_enable)
        lay.addWidget(self.top_bypass)
        return bar

    def eventFilter(self, obj, ev):
        """B = bypass, app-wide.  Handled at ShortcutOverride, which Qt delivers BEFORE handing the key to the input
        method, so it also works with 注音／拼音 input sources active (a plain KeyPress never arrives then)."""
        t = ev.type()
        if t in (QEvent.ShortcutOverride, QEvent.KeyPress) and ev.key() == Qt.Key_B and not ev.isAutoRepeat() \
                and ev.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
            fw = obj if not isinstance(obj, QApplication) else QApplication.focusWidget()
            from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit, QTextEdit
            if isinstance(fw, (QLineEdit, QTextEdit)) and not isinstance(fw.parent(), (QAbstractSpinBox, QComboBox)) \
                    and not (isinstance(fw, QTextEdit) and fw.isReadOnly()):
                return False                                  # a real text field: let the letter through
            if t == QEvent.ShortcutOverride:                 # Qt sends this first for every physical press
                self.toggle_bypass()
            ev.accept()                                       # the KeyPress (if any) is swallowed, no second toggle
            return True
        return super().eventFilter(obj, ev)

    def sync_topbar(self):
        if not hasattr(self, "top_slots"):
            return
        self.top_slots.set_value(self.slot)
        self.top_enable.set_on(self.enabled)
        on = bool(self.global_bypass)
        self.top_bypass.variant = "primary" if on else "outline"
        g = self.engine.bypass_gain_db() if self.engine.level_match else 0.0
        self.top_bypass.setText(("Bypass 中（B 恢復）" if on else "Bypass（B）") + (f"  {g:+.1f} dB" if abs(g) >= 0.05 else ""))
        self.top_bypass.setToolTip("一鍵比較原始聲音；任何頁面按鍵盤 B 也可以切換。"
                                   + (f"\nBypass 時原始聲音衰減 {g:+.1f} dB，對齊濾波器的中頻音量（公平 A/B，可在「輸出裝置」關閉）"
                                      if abs(g) >= 0.05 else ""))
        self.top_bypass.update()

    def toggle_bypass(self):
        self.set_bypass(not self.global_bypass)

    # ------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        from PySide6.QtWidgets import QFrame
        sb = QFrame()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(232)
        v = QVBoxLayout(sb)
        v.setContentsMargins(14, 20, 14, 16)
        v.setSpacing(6)
        row = QHBoxLayout()
        lg = QLabel()
        lg.setPixmap(logo_pixmap())
        lg.setFixedSize(36, 36)
        t = QLabel("roomcorr")
        t.setStyleSheet("font-size:19px;font-weight:800;")
        row.addWidget(lg)
        row.addWidget(t)
        row.addStretch(1)
        v.addLayout(row)
        v.addSpacing(14)
        self.nav = {}
        for key, icon, text in PAGES:
            b = NavButton(icon, text)
            b.clicked.connect(lambda _=False, k=key: self.goto(k))
            v.addWidget(b)
            self.nav[key] = b
        v.addStretch(1)
        self.device_card = DeviceCard()
        self.device_card.clicked.connect(self.open_audio)
        v.addWidget(self.device_card)
        return sb

    def goto(self, key):
        if key != "export" and "export" in self.pages:
            self.pages["export"].stop_listen()
        self.stack.setCurrentWidget(self.pages[key])
        for k, b in self.nav.items():
            b.setChecked(k == key)
        if key == "analysis" and self.analysis is None and self.session is not None and self.aworker is None \
                and self.session.has("L", 1):
            self.run_analysis()
        if key == "bass":
            self.pages["bass"].reload_if_visible()

    # ------------------------------------------------------------ config
    def scan_devices(self, rescan=False):
        """(Re)read the CoreAudio device list.  rescan=True re-initialises PortAudio so devices plugged in or
        switched on after launch show up."""
        import sounddevice as sd
        if rescan and not self.engine.running:
            try:
                sd._terminate()
                sd._initialize()
            except Exception as e:  # noqa: BLE001
                print("portaudio re-init failed", e)
        self.devs = sd.query_devices()
        ins = [(i, d["name"]) for i, d in enumerate(self.devs) if d["max_input_channels"] >= 2]
        outs = [(i, d["name"]) for i, d in enumerate(self.devs) if d["max_output_channels"] >= 1]
        self.play_in = ins
        self.play_out = [(i, n) for i, n in outs if self.devs[i]["max_output_channels"] >= 2]
        self.mic_devices = [d["name"] for d in self.devs if d["max_input_channels"] >= 1 and "blackhole" not in d["name"].lower()]
        self.out_devices = [n for i, n in self.play_out if "blackhole" not in n.lower()]
        self.dev_index = {d["name"]: i for i, d in enumerate(self.devs)}

    def dev_id(self, name, what="裝置"):
        """PortAudio index for a device name; rescans and falls back to a case-insensitive / prefix match."""
        if name in self.dev_index:
            return self.dev_index[name]
        self.scan_devices(rescan=not self.engine.running)
        if name in self.dev_index:
            return self.dev_index[name]
        low = (name or "").strip().lower()
        for n, i in self.dev_index.items():
            if n.strip().lower() == low or (low and (n.lower().startswith(low) or low.startswith(n.lower()))):
                return i
        raise RuntimeError(f"找不到{what}「{name}」。請確認它已開機並接上，然後到左下角「輸出裝置」按「重新掃描」重新選擇。"
                           "\n\n目前系統看到的裝置：\n" + self.device_report())

    def device_report(self):
        lines = []
        for i, d in enumerate(self.devs):
            lines.append(f"{d['name']}　輸入 {d['max_input_channels']} 聲道／輸出 {d['max_output_channels']} 聲道"
                         f"　{d['default_samplerate'] / 1000:g} kHz")
        return "\n".join(lines) if lines else "（沒有找到任何音訊裝置）"

    def load_cfg(self):
        c = self.cfg
        self.scan_devices()
        ins, outs = self.play_in, self.play_out

        def first(lst, words):
            return next((n for n in lst if any(w in n.lower() for w in words)), None)
        saved = json.loads(c.value("acfg", "{}") or "{}")
        a_in = saved.get("in") or first([n for _, n in ins], ("blackhole",)) or (ins[0][1] if ins else "")
        a_out = saved.get("out") or next((n for n in self.out_devices), "")
        self.acfg = dict(rate=48000, block=1024, trim=0, match=True)
        self.acfg.update(saved)
        self.acfg.update({"in": a_in, "out": a_out})
        saved = json.loads(c.value("mcfg", "{}") or "{}")
        m_in = saved.get("m_in") or first(self.mic_devices, ("umik", "mic", "minidsp", "usb"))
        if not m_in:
            m_in = next((d["name"] for d in self.devs if d["max_input_channels"] >= 1 and d["max_output_channels"] == 0
                         and "blackhole" not in d["name"].lower()), self.mic_devices[0] if self.mic_devices else "")
        self.mcfg = dict(m_in=m_in, m_out=a_out, mic_ch=1, lout=1, rout=2, rate=48000, dur=8, level=-24, rep=1,
                         nsub=0, npos=5, subch=[3, 4, 5, 6])
        self.mcfg.update(saved)
        self.mcfg["m_in"] = saved.get("m_in") or m_in
        self.bass_cfg = dict(fc=80.0, fam="linkwitz-riley", order=4, phase="linear", method="mso", main="design",
                             eq=True, max_delay=20.0, channels=[1, 2, 3, 4, 5, 6])
        self.bass_cfg.update(json.loads(c.value("bass_cfg", "{}") or "{}"))
        self.export_cfg = dict(folder=DEFAULT_FOLDER, ask=True)
        self.export_cfg.update(json.loads(c.value("export_cfg", "{}") or "{}"))
        saved = json.loads(c.value("slot_settings", "{}") or "{}")
        self.settings = {sid: A.slot_settings(sid, saved.get(sid)) for sid, _ in A.SLOTS}
        self.slot = c.value("slot", "std")
        if self.slot not in self.settings:
            self.slot = "std"
        self.enabled = c.value("enabled", False, bool)
        self.mic_cal = c.value("mic_cal", "") or None

    def save_cfg(self):
        c = self.cfg
        c.setValue("acfg", json.dumps(self.acfg))
        c.setValue("mcfg", json.dumps(self.mcfg))
        c.setValue("bass_cfg", json.dumps(self.bass_cfg))
        c.setValue("export_cfg", json.dumps(self.export_cfg))
        c.setValue("slot_settings", json.dumps(self.settings))
        c.setValue("slot", self.slot)
        c.setValue("enabled", self.enabled)
        c.setValue("mic_cal", self.mic_cal or "")
        c.setValue("session", self.session.name if self.session else "")

    @property
    def cur(self):
        return self.settings[self.slot]

    def set_mcfg(self, k, v):
        self.mcfg[k] = v
        self.save_cfg()

    def set_export(self, k, v):
        self.export_cfg[k] = v
        self.save_cfg()
        self.stateChanged.emit()

    def audio_rate(self):
        return int(self.acfg["rate"])

    def status(self, msg):
        self.statusBar().showMessage(msg, 6000)

    # ------------------------------------------------------------ slot / settings
    def update_setting(self, key, value, clear_points=False, preset=None):
        s = self.cur
        s[key] = value
        if clear_points:
            s["points"] = None
        if preset:
            s["preset"] = preset
        self.save_cfg()
        self.settingsChanged.emit()
        self.schedule_design()

    def update_many(self, d):
        self.cur.update(d)
        self.save_cfg()
        self.settingsChanged.emit()
        self.schedule_design()

    def set_slot(self, sid):
        if sid == self.slot:
            return
        self.slot = sid
        self.save_cfg()
        self.apply_selection()
        self.settingsChanged.emit()
        self.stateChanged.emit()
        self.schedule_design()

    def slot_name(self, sid=None):
        return dict(A.SLOTS)[sid or self.slot]

    def slot_folder(self, sid=None):
        return os.path.join(PROFILE_DIR, self.slot_name(sid))

    def refresh_all(self):
        self.update_device_card()
        self.update_tray()
        self.measureChanged.emit()
        self.stateChanged.emit()

    def update_tray(self):
        self.tray.setIcon(dot_icon(COBALT if self.enabled else "#9AA5BB"))
        self.tray.setToolTip("roomcorr：" + (f"校正 開（{self.slot_name()}）" if self.enabled else "校正 關"))
        self.a_toggle.setText("校正 關閉" if self.enabled else "校正 開啟")
        for sid, a in self.slot_actions.items():
            a.setChecked(sid == self.slot)

    def update_device_card(self):
        self.device_card.name.setText(self.acfg["out"] or "（未選擇）")
        run = self.engine.running
        self.device_card.set_status("已連接" if run else "未啟動", run)

    def show_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------ profiles / engine
    def reload_profiles_and_apply(self):
        self.reload_profiles()
        self.apply_selection()
        self.stateChanged.emit()

    def after_profile_change(self):
        self.reload_profiles_and_apply()

    def set_enabled(self, v):
        if v and self.slot not in self.profiles:
            self.status(f"「{self.slot_name()}」還沒有匯出濾波器：請先到「濾波器」或「匯出」按「套用並匯出」。")
        self.enabled = bool(v)
        if self.enabled and not self.engine.running and not self.measuring:
            self.start_audio()                      # turning correction on should make it audible
        self.apply_selection()
        self.save_cfg()
        self.update_tray()
        self.stateChanged.emit()

    def apply_selection(self):
        prof = self.profiles.get(self.slot) if self.enabled else None
        if prof is not None and prof.n_out > self.engine.n_out and self.engine.running:
            return self.start_audio()
        ref = prof
        if getattr(self, "preview_prof", None) is not None:
            ref = prof = self.preview_prof
        elif getattr(self, "preview_bypass", False):
            prof = None
        elif getattr(self, "audition_h", None) is not None:
            ref = prof = self._audition_profile(prof, self.audition_h)
            if getattr(self, "audition_bypass", False):
                prof = None
        if getattr(self, "global_bypass", False):
            prof = None
        self.engine.select(prof, ref)
        self.update_tray()
        self.sync_topbar()

    def _audition_profile(self, base, h):
        """Temporary profile = current output filters convolved with the FIR-tool chain (live audition).
        h: one FIR for both channels, or {"L": hL, "R": hR}."""
        fs = self.audio_rate()
        hh = h if isinstance(h, dict) else {"L": h, "R": h}
        if base is None:
            return Profile("試聽 FIR", hh["L"], hh["R"], fs)
        outs = []
        for o in base.outputs:
            hc = hh.get(o["name"])
            paths = [(inp, signal.fftconvolve(np.asarray(x, float), hc) if hc is not None else x, g) for inp, x, g in o["paths"]]
            outs.append(dict(name=o["name"], channel=o["channel"], paths=paths))
        return Profile("試聽 FIR", None, None, fs, outputs=outs)

    def set_preview(self, on, bypass=False):
        """Export page: play the currently designed (not yet saved) filter through the engine."""
        ds, an = self.design_result, self.analysis
        if not on or ds is None:
            self.preview_prof, self.preview_bypass = None, False
            self.apply_selection()
            return
        fs = self.audio_rate()
        hl, hr = (self._resample(np.asarray(ds.h[c], float), an.fs, fs) for c in "LR")
        self.preview_prof = None if bypass else Profile("預覽", hl, hr, fs)
        self.preview_bypass = bypass
        if not self.engine.running and not self.measuring:
            self.start_audio()
        self.engine.select(self.preview_prof)
        self.update_tray()

    def set_bypass(self, on):
        """One-click bypass of whatever is playing (saved correction, unsaved preview or FIR-tool audition)."""
        self.global_bypass = bool(on)
        if on and not self.engine.running and not self.measuring:
            self.start_audio()
        self.apply_selection()
        self.status("Bypass：原始聲音（按 B 恢復）" if on else "已恢復校正")
        self.stateChanged.emit()

    def set_audition(self, h="keep", bypass=None):
        """Called by the FIR tool: h = chain FIR to audition (None = stop, "keep" = unchanged); bypass = A/B switch."""
        if not isinstance(h, str):
            self.audition_h = h
        if bypass is not None:
            self.audition_bypass = bool(bypass)
        if (self.audition_h is not None) and not self.engine.running and not self.measuring:
            self.start_audio()
        self.apply_selection()
        self.status("FIR 工具：" + ("Bypass" if self.audition_bypass else ("這條鏈正即時套用" if self.audition_h is not None else "未套用")))

    def start_audio(self):
        if self.measuring or self.mworker is not None:
            return
        try:
            a = self.acfg
            if not a["in"] or not a["out"]:
                raise RuntimeError("請先在左下角「輸出裝置」選擇輸入（BlackHole）與輸出（DAC）。")
            if a["in"] == a["out"] or "blackhole" in a["out"].lower():
                raise RuntimeError("輸出必須是你的 DAC／喇叭，不能是輸入裝置或 BlackHole（會造成回授）。")
            self.engine.stop()
            in_id, out_id = self.dev_id(a["in"], "輸入裝置"), self.dev_id(a["out"], "輸出裝置")
            self.reload_profiles()
            n_out = max([2] + [p.n_out for p in self.profiles.values()])
            have = self.devs[out_id]["max_output_channels"]
            if n_out > have:
                raise RuntimeError(f"設定檔需要 {n_out} 個輸出聲道，但「{a['out']}」只有 {have} 個。"
                                   "請改用多聲道 DAC／音訊介面，或移除含低音砲的設定檔。")
            self.engine.configure(int(a["rate"]), int(a["block"]), n_out)
            self.engine._want = None
            self.engine._cur = None
            self.engine.level_match = a["match"]
            self.engine.gain_db = a["trim"] / 10
            self.apply_selection()
            self.engine.start(in_id, out_id)
            self.save_cfg()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "音訊錯誤", str(e))
        self.update_device_card()

    def open_audio(self):
        AudioDialog(self).exec()
        self.update_device_card()

    def open_measure_settings(self):
        MeasureDialog(self).exec()

    def open_advanced(self):
        AdvancedDialog(self).exec()

    def open_fir_tool(self):
        d = QDialog(self)
        d.setWindowTitle("FIR 工具")
        v = QVBoxLayout(d)
        v.setContentsMargins(12, 12, 12, 12)
        tab = FirToolTab(self)
        v.addWidget(tab)
        d.resize(1380, 920)
        d.finished.connect(lambda *_: tab.stop_audition())
        self.fir_dialog = d
        d.exec()
        self.fir_dialog = None

    def reload_profiles(self, *a):
        self.profiles = {}
        fs = int(self.acfg["rate"])
        for sid, name in A.SLOTS:
            p = self.slot_folder(sid)
            if os.path.isdir(p):
                try:
                    self.profiles[sid] = Profile.from_folder(p, fs)
                except Exception as e:  # noqa: BLE001
                    print("skip profile", name, e)

    # ------------------------------------------------------------ measurement
    def pick_cal(self):
        p, _ = QFileDialog.getOpenFileName(self, "麥克風校正檔（頻率 dB 兩欄文字檔）", "", "Text (*.txt *.cal *.csv);;All (*)")
        if p:
            try:
                R.load_curve(p)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "無法讀取", str(e))
                return
            self.mic_cal = p
            self.save_cfg()
            self.measureChanged.emit()

    def sweep_args(self, kind):
        m = self.mcfg
        outs = {"L": [m["lout"]], "R": [m["rout"]], "LR": [m["lout"], m["rout"]]}.get(kind) \
            or [m["subch"][int(kind[1:]) - 1]]
        return dict(fs=int(m["rate"]), mic_dev=self.dev_id(m["m_in"], "麥克風輸入裝置"), mic_ch=int(m["mic_ch"]),
                    out_dev=self.dev_id(m["m_out"], "量測輸出裝置"), out_chs=outs)

    def _pause_engine(self):
        self.engine_was_running = self.engine.running
        self.engine.stop()
        self.update_device_card()

    def _resume_engine(self):
        if self.engine_was_running:
            self.engine_was_running = False
            self.start_audio()

    def do_level_check(self):
        if self.mworker is not None:
            return
        if QMessageBox.warning(self, "測試音量", "將從左右喇叭播放 2 秒掃頻。請先把放大器音量調低。繼續？",
                               QMessageBox.Ok | QMessageBox.Cancel) != QMessageBox.Ok:
            return
        try:
            a = self.sweep_args("LR")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "裝置", str(e))
            return
        self._pause_engine()
        lvl = self.mcfg["level"]
        self.pages["measure"].lb_level.setText("測試中…")
        self.mworker = Worker(lambda prog: level_check(a["fs"], a["mic_dev"], a["mic_ch"], a["out_dev"], a["out_chs"], lvl))
        self.mworker.done.connect(self._level_done)
        self.mworker.failed.connect(self._level_failed)
        self.mworker.start()

    def _level_done(self, peak):
        self.mworker = None
        hint = "太大聲，請降低" if peak > -6 else ("太小聲，請提高" if peak < -45 else "OK")
        self.pages["measure"].lb_level.setText(f"麥克風峰值 {peak:.0f} dBFS（{hint}）")
        self._resume_engine()

    def _level_failed(self, msg):
        self.mworker = None
        self.pages["measure"].lb_level.setText("")
        QMessageBox.warning(self, "測試失敗", msg)
        self._resume_engine()

    def ensure_session(self):
        if self.session is None:
            self.new_session()
        return self.session

    def new_session(self):
        self.session = Session(os.path.join(SESS_DIR, time.strftime("%Y%m%d-%H%M%S")))
        self.analysis = None
        self.design_result = None
        self.save_cfg()
        self.measureChanged.emit()

    def start_measure(self, only=None):
        if self.measuring or self.mworker is not None:
            return
        s = self.ensure_session()
        every = tasks_for(self.mcfg["nsub"], self.mcfg["npos"])
        queue = [x for x in every if (x == only if only else not s.has(*x))]
        if only is None and not queue:
            if QMessageBox.question(self, "重新測量", "所有項目都已測量。要全部重測嗎？") != QMessageBox.Yes:
                return
            for x in every:
                s.delete(*x)
            queue = every
        if QMessageBox.warning(self, "開始測量", "掃頻會以目前放大器音量播放。請先把音量調低，並讓房間保持安靜。繼續？",
                               QMessageBox.Ok | QMessageBox.Cancel) != QMessageBox.Ok:
            return
        self._pause_engine()
        self.mqueue, self.m_total, self.m_done, self.mcur_pos, self.m_aborted = queue, len(queue), 0, None, False
        self.measuring = True
        self.measureChanged.emit()
        self._next_task()

    def _next_task(self):
        if self.m_aborted or not self.mqueue:
            return self._finish_measure()
        kind, pos = self.mqueue[0]
        self.current_task = (kind, pos)
        desc = pos_name(pos)
        if pos != self.mcur_pos:
            r = QMessageBox.information(self, "移動麥克風", f"請把麥克風放到：\n\n    {desc}\n\n（耳朵高度）放好後按 OK 開始。",
                                        QMessageBox.Ok | QMessageBox.Cancel)
            if r != QMessageBox.Ok:
                self.m_aborted = True
                return self._finish_measure()
            self.mcur_pos = pos
        try:
            a = self.sweep_args(kind)
        except Exception as e:  # noqa: BLE001
            self.m_aborted = True
            QMessageBox.warning(self, "裝置", str(e))
            return self._finish_measure()
        m = self.mcfg
        dur, lvl, rep = int(m["dur"]), int(m["level"]), int(m["rep"])
        pg = self.pages["measure"]
        pg.sel = (kind, pos)
        pg.set_progress(self.m_done / self.m_total,
                        f"[{self.m_done + 1}/{self.m_total}] {desc} · {kind_name(kind)}　掃頻中（約 {(dur + 2) * rep:.0f} 秒）…")
        self.measureChanged.emit()
        self.mworker = Worker(lambda prog: run_sweep(a["fs"], a["mic_dev"], a["mic_ch"], a["out_dev"], a["out_chs"],
                                                     dur, lvl, rep))
        self.mworker.done.connect(self._task_done)
        self.mworker.failed.connect(self._task_failed)
        self.mworker.start()

    def _task_done(self, res):
        self.mworker = None
        if self.m_aborted:
            return self._finish_measure()
        full, peak, snr = res
        kind, pos = self.mqueue[0]
        if peak > -0.5:
            self.m_aborted = True
            QMessageBox.warning(self, "麥克風輸入削波", "麥克風輸入過大（削波）。請降低掃頻音量或麥克風增益後重測。")
            return self._finish_measure()
        self.mqueue.pop(0)
        self.session.save(kind, pos, full, int(self.mcfg["rate"]), snr)
        self.m_done += 1
        self.pages["measure"].set_progress(self.m_done / self.m_total, f"完成 {pos_name(pos)} · {kind_name(kind)}")
        if isinstance(snr, (int, float)) and snr == snr and snr < 30:
            self.pages["measure"].status.setText(f"⚠ {pos_name(pos)} · {kind_name(kind)} 的 SNR 只有 {snr:.0f} dB，建議提高音量或減少噪音後重測。")
        self.measureChanged.emit()
        QTimer.singleShot(250, self._next_task)

    def _task_failed(self, msg):
        self.mworker = None
        self.m_aborted = True
        QMessageBox.warning(self, "測量失敗", msg)
        self._finish_measure()

    def abort_measure(self):
        self.m_aborted = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:  # noqa: BLE001
            pass

    def _finish_measure(self):
        if self.mworker is not None:
            return
        self.measuring = False
        self.current_task = None
        pg = self.pages["measure"]
        self._resume_engine()
        s = self.session
        ready = s is not None and any(s.has("L", p) for p in (1, 2, 3)) and any(s.has("R", p) for p in (1, 2, 3))
        if self.m_aborted:
            pg.set_progress(0, "已中止。")
        else:
            pg.set_progress(1.0, "測量完成。")
        self.measureChanged.emit()
        if ready and not self.m_aborted:
            self.analysis = None
            self.goto("analysis")
            self.run_analysis()

    # ------------------------------------------------------------ analysis / design
    def run_analysis(self):
        if self.aworker is not None:
            return
        s = self.session
        if s is None:
            return
        meas, both, fs = s.load()
        if not meas["L"] or not meas["R"]:
            self.status("至少需要左、右喇叭各一個位置的測量。")
            return
        self.status("分析中…")
        cal = self.mic_cal
        self.aworker = Worker(lambda prog: A.Analysis(meas, both, fs, cal, progress=prog))
        self.aworker.progress.connect(self.status)
        self.aworker.done.connect(self._analysis_done)
        self.aworker.failed.connect(self._analysis_failed)
        self.aworker.start()

    def _analysis_done(self, an):
        self.aworker = None
        self.analysis = an
        self.design_result = None
        self.status("分析完成")
        self.analysisChanged.emit()
        if getattr(self, "_auto_pending", False):
            self._auto_pending = False
            self.auto_optimize()
            return
        self.run_design()

    def _analysis_failed(self, msg):
        self.aworker = None
        QMessageBox.warning(self, "分析失敗", msg)

    def schedule_design(self):
        if self.analysis is not None:
            self.design_timer.start(250)

    def run_design(self):
        if self.analysis is None:
            return
        if self.dworker is not None:
            self.d_dirty = True
            return
        p = A.params_from_settings(self.cur)
        an = self.analysis
        self.dworker = Worker(lambda prog: an.design(p))
        self.dworker.done.connect(self._design_done)
        self.dworker.failed.connect(self._design_failed)
        self.dworker.start()

    def _design_done(self, ds):
        self.dworker = None
        self.design_result = ds
        self.designChanged.emit()
        if self.d_dirty:
            self.d_dirty = False
            self.run_design()

    def _design_failed(self, msg):
        self.dworker = None
        QMessageBox.warning(self, "計算失敗", msg)

    # ------------------------------------------------------------ export
    def export_file_names(self):
        pre = {"std": "std", "night": "night", "movie": "movie"}[self.slot]
        names = [f"{pre}_L.wav", f"{pre}_R.wav"]
        bp = self.pages.get("bass") if hasattr(self, "pages") else None
        if bp is not None and bp.applied:
            names += [f"{pre}_{nm}.wav" for nm, _ in bp.applied["G"]]
        return names

    @staticmethod
    def _resample(h, fs_from, fs_to):
        if fs_from == fs_to:
            return h
        g = gcd(int(fs_from), int(fs_to))
        return signal.resample_poly(h, int(fs_to) // g, int(fs_from) // g)

    def auto_optimize(self):
        """One button: derive settings from the measurement, pick the best candidate, save into the slot, enable."""
        if self.analysis is None:
            if self.session is None:
                QMessageBox.information(self, "還沒有量測", "請先到「量測」完成左右喇叭的量測，再按自動最佳化。")
                return
            self.run_analysis()
            self._auto_pending = True
            return
        if self.aworker is not None or getattr(self, "oworker", None) is not None:
            return
        an, s0 = self.analysis, dict(self.cur)
        cons = float(self.cfg.value("auto_conservative", 0.5))
        self.status("自動最佳化中…")
        self.oworker = Worker(lambda prog: A.auto_settings(an, s0, cons, prog))
        self.oworker.progress.connect(self.status)
        self.oworker.done.connect(self._auto_done)
        self.oworker.failed.connect(lambda m: (setattr(self, "oworker", None), QMessageBox.warning(self, "自動最佳化失敗", m)))
        self.oworker.start()

    def _auto_done(self, res):
        self.oworker = None
        cand, lines, ds = res
        self.cur.update(cand)
        self.save_cfg()
        self.design_result = ds
        self.settingsChanged.emit()
        self.designChanged.emit()
        self.export_current(quiet=True)
        self.set_enabled(True)
        self.goto("analysis")
        QMessageBox.information(self, "自動最佳化完成",
                                "\n".join(lines) + f"\n\n濾波器已存入「{self.slot_name()}」並開啟校正。"
                                "可到「分析」頁比較曲線，或用 REW 重新量測驗證。")

    def export_current(self, quiet=False):
        ds, an = self.design_result, self.analysis
        if ds is None or an is None:
            QMessageBox.information(self, "還沒有濾波器", "請先完成量測與分析。")
            return
        s = self.cur
        pre = {"std": "std", "night": "night", "movie": "movie"}[self.slot]
        outs, info = None, None
        bp = self.pages["bass"]
        if bp.applied:
            main_h = {c: ds.h[c] / ds.scale for c in "LR"} if self.bass_cfg["main"] == "design" else None
            outs, info = bp.build_outputs(main_h, int(s["taps"]))
            n = 1 << 17
            from scipy.fft import rfft
            mx = max(float(np.abs(sum(rfft(o["h"], n) * g for _, g in o["paths"])).max()) for o in outs)
            sc = min(1.0, 10 ** (-1.0 / 20) / mx)
            for o in outs:
                o["h"] = o["h"] * sc
            files = {f"{pre}_{o['name']}.wav": o["h"] for o in outs}
        else:
            files = {f"{pre}_L.wav": ds.h["L"], f"{pre}_R.wav": ds.h["R"]}
        out_fs = int(s["out_fs"])
        folder = self.export_cfg["folder"]
        exist = [n for n in files if os.path.exists(os.path.join(folder, n))]
        if self.export_cfg["ask"] and not quiet:
            txt = f"將寫入資料夾：\n{folder}\n\n" + "\n".join(files) + f"\n\n取樣率 {out_fs / 1000:g} kHz；同一組濾波器也會存成「{self.slot_name()}」設定檔。"
            if exist:
                txt += "\n\n⚠ 以下檔案已存在，會被覆蓋：\n" + "\n".join(exist)
            if QMessageBox.question(self, "確認儲存", txt) != QMessageBox.Yes:
                return
        try:
            os.makedirs(folder, exist_ok=True)
            for n, h in files.items():
                wavfile.write(os.path.join(folder, n), out_fs, self._resample(np.asarray(h, float), an.fs, out_fs).astype(np.float32))
            sf = self.slot_folder()
            if outs:
                D.save_multi_profile(sf, an.fs, outs)
            else:
                os.makedirs(sf, exist_ok=True)
                for ch in "LR":
                    wavfile.write(os.path.join(sf, f"{ch}.wav"), an.fs, np.asarray(ds.h[ch], np.float32))
                pj = os.path.join(sf, "profile.json")
                if os.path.exists(pj):
                    os.remove(pj)
            with open(os.path.join(sf, "settings.json"), "w") as fh:
                json.dump(self.cur, fh, ensure_ascii=False, indent=1)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "儲存失敗", str(e))
            return
        self.pages["export"].stop_listen()
        self.reload_profiles()
        self.apply_selection()
        self.status(f"已匯出 {len(files)} 個檔案到 {folder}，並更新「{self.slot_name()}」設定檔")
        self.stateChanged.emit()

    # ------------------------------------------------------------ window
    def closeEvent(self, e):
        self.save_cfg()
        if self.tray.isVisible() and self.engine.running:
            self.hide()
            e.ignore()
        else:
            self.engine.stop()
            e.accept()


def _install_crash_log():
    import faulthandler
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log = open(LOG_FILE, "a", buffering=1)
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start =====\n")
    faulthandler.enable(file=log)

    def hook(et, ev, tb):
        log.write("".join(traceback.format_exception(et, ev, tb)))
        try:
            QMessageBox.critical(None, "roomcorr 發生錯誤", f"{ev}\n\n詳細內容已記錄在：\n{LOG_FILE}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook
    return log


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("roomcorr")
    _install_crash_log()
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(lambda: w.engine.stop())
    w = MainWindow()
    w.resize(1240, 880)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
