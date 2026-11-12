"""roomcorr_theme.py - white / blue-morpho visual theme and custom widgets (matches the roomcorr design deck)."""
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient)
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

# palette ----------------------------------------------------------------
BG = "#F6F8FC"
SIDEBAR = "#F1F5FB"
CARD_BORDER = "#E4E9F3"
TEXT = "#0F1A2B"
MUTED = "#55617A"
FAINT = "#8A94A8"
CYAN = "#0B7CC0"
BRIGHT = "#2AA8F0"
COBALT = "#1456D1"
DEEP = "#0A37A6"
ORANGE = "#E8820C"
NAVY = "#1B2A4A"
FONT = "PingFang TC, Noto Sans TC, Helvetica Neue, Arial"

STYLE = f"""
* {{ font-family: {FONT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 16px; }}
QFrame#sidebar {{ background: {SIDEBAR}; border: none; border-right: 1px solid {CARD_BORDER}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: #FFFFFF; border: 1px solid #D6DDEC; border-radius: 9px; padding: 6px 10px; min-height: 20px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {BRIGHT}; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; border: 1px solid #D6DDEC; selection-background-color: {COBALT}; selection-color: white; }}
QTextEdit {{ background: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 10px; padding: 6px; }}
QListWidget, QTableWidget {{ background: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 10px; }}
QListWidget::item {{ padding: 8px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: {COBALT}; color: white; }}
QHeaderView::section {{ background: {SIDEBAR}; border: 0; padding: 6px; color: {MUTED}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid #C5CEE0; background: white; }}
QCheckBox::indicator:checked {{ background: {COBALT}; border-color: {COBALT}; }}
QSlider::groove:horizontal {{ height: 6px; background: #E3E8F2; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {CYAN}, stop:1 {COBALT}); border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {COBALT}; border: 3px solid white; width: 14px; height: 14px; margin: -8px 0; border-radius: 10px; }}
QProgressBar {{ background: #E8EDF6; border: none; border-radius: 5px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {BRIGHT}, stop:1 {COBALT}); border-radius: 5px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #CBD3E4; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QMessageBox {{ background: #FFFFFF; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 4px 8px; }}
"""


def lighten(c, f=1.12):
    return QColor(c).lighter(int(f * 100))


def _glow(p, rect, strength=90):
    g = QRadialGradient(QPointF(rect.right() - rect.width() * 0.18, rect.top() - 2), rect.width() * 0.42)
    g.setColorAt(0, QColor(255, 255, 255, strength))
    g.setColorAt(1, QColor(255, 255, 255, 0))
    p.setBrush(QBrush(g))
    p.setPen(Qt.NoPen)
    p.drawRect(rect)


def paint_gradient_rect(p, rect, radius, deep=False, glow=True):
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    if deep:
        g.setColorAt(0, QColor("#1560D8"))
        g.setColorAt(0.55, QColor(COBALT))
        g.setColorAt(1, QColor(DEEP))
    else:
        g.setColorAt(0, QColor("#1560D8"))
        g.setColorAt(0.5, QColor("#0E78C8"))
        g.setColorAt(1, QColor("#1646C4"))
    p.fillPath(path, QBrush(g))
    if glow:
        p.save()
        p.setClipPath(path)
        _glow(p, rect)
        p.restore()


# ------------------------------------------------------------------ icons
def nav_icon(name, color, size=20):
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(2, 2)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = size
    if name == "mic":
        p.drawRoundedRect(QRectF(s * .38, s * .12, s * .24, s * .42), s * .12, s * .12)
        path = QPainterPath()
        path.moveTo(s * .24, s * .46)
        path.arcTo(QRectF(s * .24, s * .30, s * .52, s * .44), 180, 180)
        p.drawPath(path)
        p.drawLine(QPointF(s * .5, s * .74), QPointF(s * .5, s * .88))
        p.drawLine(QPointF(s * .38, s * .88), QPointF(s * .62, s * .88))
    elif name == "chart":
        p.drawLine(QPointF(s * .15, s * .12), QPointF(s * .15, s * .85))
        p.drawLine(QPointF(s * .15, s * .85), QPointF(s * .88, s * .85))
        p.drawPolyline([QPointF(s * .28, s * .66), QPointF(s * .47, s * .45), QPointF(s * .62, s * .58),
                        QPointF(s * .84, s * .26)])
    elif name == "wave":
        path = QPainterPath()
        path.moveTo(s * .1, s * .5)
        path.cubicTo(s * .25, s * .1, s * .35, s * .1, s * .5, s * .5)
        path.cubicTo(s * .65, s * .9, s * .75, s * .9, s * .9, s * .5)
        p.drawPath(path)
    elif name == "sliders":
        for i, (x, k) in enumerate(((.25, .35), (.5, .65), (.75, .45))):
            p.drawLine(QPointF(s * x, s * .15), QPointF(s * x, s * .85))
            p.drawLine(QPointF(s * (x - .08), s * k), QPointF(s * (x + .08), s * k))
    elif name == "pulse":
        p.drawPolyline([QPointF(s * .08, s * .5), QPointF(s * .3, s * .5), QPointF(s * .4, s * .22),
                        QPointF(s * .55, s * .78), QPointF(s * .66, s * .5), QPointF(s * .92, s * .5)])
    elif name == "download":
        p.drawLine(QPointF(s * .5, s * .15), QPointF(s * .5, s * .62))
        p.drawPolyline([QPointF(s * .32, s * .46), QPointF(s * .5, s * .64), QPointF(s * .68, s * .46)])
        p.drawLine(QPointF(s * .2, s * .84), QPointF(s * .8, s * .84))
    elif name == "logo":
        p.setPen(QPen(QColor("white"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolyline([QPointF(s * .12, s * .5), QPointF(s * .3, s * .5), QPointF(s * .4, s * .24),
                        QPointF(s * .54, s * .76), QPointF(s * .64, s * .5), QPointF(s * .88, s * .5)])
    p.end()
    pm.setDevicePixelRatio(2)
    return pm


# ------------------------------------------------------------------ basic building blocks
class Card(QFrame):
    def __init__(self, title="", subtitle=""):
        super().__init__()
        self.setObjectName("card")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(20, 18, 20, 18)
        self.v.setSpacing(10)
        if title:
            t = QLabel(title)
            t.setStyleSheet("font-size:16px;font-weight:700;")
            self.v.addWidget(t)
        if subtitle:
            st = QLabel(subtitle)
            st.setStyleSheet(f"color:{MUTED};")
            self.v.addWidget(st)


def h_title(title, subtitle=""):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    t = QLabel(title)
    t.setStyleSheet("font-size:26px;font-weight:800;")
    lay.addWidget(t)
    s = QLabel(subtitle)
    s.setStyleSheet(f"color:{MUTED};font-size:13px;")
    lay.addWidget(s)
    w.title_label, w.sub_label = t, s
    return w


class GradientButton(QPushButton):
    def __init__(self, text, variant="primary"):
        super().__init__(text)
        self.variant = variant
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(40)
        self.setFont(QFont("PingFang TC", 13, QFont.Bold))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if not self.isEnabled():
            p.setBrush(QColor("#E3E8F2"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 11, 11)
            p.setPen(QColor(FAINT))
        elif self.variant == "primary":
            paint_gradient_rect(p, r, 11)
            if self.underMouse():
                p.setBrush(QColor(255, 255, 255, 28))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(r, 11, 11)
            p.setPen(QColor("white"))
        else:
            p.setBrush(QColor("#F3F6FC") if self.underMouse() else QColor("white"))
            p.setPen(QPen(QColor("#D6DDEC"), 1.2))
            p.drawRoundedRect(r, 11, 11)
            p.setPen(QColor(TEXT))
        p.setFont(self.font())
        p.drawText(r, Qt.AlignCenter, self.text())


class Segmented(QWidget):
    changed = Signal(object)

    def __init__(self, items, value=None, small=False):
        super().__init__()
        self.setObjectName("seg")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("QWidget#seg{background:#EEF2F9;border-radius:11px;}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.btns = {}
        for label, val in items:
            b = _SegButton(label, small)
            b.setCheckable(True)
            self.group.addButton(b)
            lay.addWidget(b)
            self.btns[val] = b
            b.clicked.connect(lambda _=False, v=val: self.changed.emit(v))
        self.set_value(value if value is not None else items[0][1], emit=False)

    def set_value(self, v, emit=False):
        if v in self.btns:
            self.btns[v].setChecked(True)
        if emit:
            self.changed.emit(v)

    def value(self):
        for v, b in self.btns.items():
            if b.isChecked():
                return v


class _SegButton(QPushButton):
    def __init__(self, text, small=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(30 if small else 34)
        self.setMinimumWidth(54 if small else 72)
        self.setFont(QFont("PingFang TC", 12 if small else 13, QFont.DemiBold))

    def sizeHint(self):
        return QSize(max(self.minimumWidth(), self.fontMetrics().horizontalAdvance(self.text()) + 30), self.minimumHeight())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        if self.isChecked():
            paint_gradient_rect(p, r, 9, glow=False)
            p.setPen(QColor("white"))
        else:
            p.setPen(QColor(MUTED))
        p.setFont(self.font())
        p.drawText(r, Qt.AlignCenter, self.text())


class Toggle(QAbstractButton):
    def __init__(self, on=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(on)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(46, 26)
        self.on_banner = False

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 0, 46, 26)
        if self.on_banner and self.isChecked():
            p.setBrush(QColor("white"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 13, 13)
            knob = QColor(COBALT)
        else:
            if self.isChecked():
                paint_gradient_rect(p, r, 13, glow=False)
            else:
                p.setBrush(QColor("#CBD3E4"))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(r, 13, 13)
            knob = QColor("white")
        x = 23 if self.isChecked() else 3
        p.setBrush(knob)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(x, 3, 20, 20))


class CapsuleToggle(QWidget):
    toggled = Signal(bool)

    def __init__(self, text_on="校正 開", text_off="校正 關"):
        super().__init__()
        self.t_on, self.t_off = text_on, text_off
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 18, 6)
        lay.setSpacing(10)
        self.sw = Toggle()
        self.sw.clicked.connect(self._c)
        self.lb = QLabel(text_off)
        self.lb.setStyleSheet("font-weight:800;font-size:14px;")
        lay.addWidget(self.sw)
        lay.addWidget(self.lb)
        self.setCursor(Qt.PointingHandCursor)

    def _c(self, v):
        self.lb.setText(self.t_on if v else self.t_off)
        self.toggled.emit(v)

    def mousePressEvent(self, e):
        self.sw.click()

    def set_on(self, v):
        self.sw.blockSignals(True)
        self.sw.setChecked(v)
        self.sw.blockSignals(False)
        self.lb.setText(self.t_on if v else self.t_off)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor("white"))
        p.setPen(QPen(QColor(COBALT if self.sw.isChecked() else "#C5CEE0"), 1.6))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)


class ToggleBanner(QWidget):
    """Blue gradient bar with title / subtitle and a white toggle (e.g. MIMO, 衰減控制)."""
    toggled = Signal(bool)

    def __init__(self, title, sub, on=False):
        super().__init__()
        self.setMinimumHeight(60)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.t = QLabel(title)
        self.t.setStyleSheet("color:white;font-weight:800;font-size:14px;")
        self.s = QLabel(sub)
        self.s.setStyleSheet("color:rgba(255,255,255,0.85);font-size:12px;")
        col.addWidget(self.t)
        col.addWidget(self.s)
        lay.addLayout(col, 1)
        self.sw = Toggle(on)
        self.sw.on_banner = True
        self.sw.clicked.connect(self._c)
        lay.addWidget(self.sw)
        self._styled = None
        self._restyle()

    def _restyle(self):
        # Never restyle inside paintEvent: setStyleSheet re-polishes -> layout -> paint, an endless loop (100 % CPU).
        on = self.sw.isChecked()
        if on == self._styled:
            return
        self._styled = on
        self.t.setStyleSheet(f"color:{'white' if on else TEXT};font-weight:800;font-size:14px;")
        self.s.setStyleSheet(f"color:{'rgba(255,255,255,0.85)' if on else MUTED};font-size:12px;")

    def _c(self, v):
        self._restyle()
        self.update()
        self.toggled.emit(v)

    def set_on(self, v):
        self.sw.blockSignals(True)
        self.sw.setChecked(v)
        self.sw.blockSignals(False)
        self._restyle()
        self.update()

    def isChecked(self):
        return self.sw.isChecked()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        if self.sw.isChecked():
            paint_gradient_rect(p, r, 12, deep=True)
        else:
            p.setBrush(QColor("#F3F6FC"))
            p.setPen(QPen(QColor("#D6DDEC"), 1))
            p.drawRoundedRect(r.adjusted(.5, .5, -.5, -.5), 12, 12)


class RadioCard(QAbstractButton):
    def __init__(self, title, sub):
        super().__init__()
        self.title, self.sub = title, sub
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = self.isChecked()
        if on:
            paint_gradient_rect(p, r, 12, deep=True)
        else:
            p.setBrush(QColor("#FAFBFE") if not self.underMouse() else QColor("#F3F6FC"))
            p.setPen(QPen(QColor("#DDE3F0"), 1.2))
            p.drawRoundedRect(r, 12, 12)
        p.setPen(QColor("white" if on else TEXT))
        f = QFont("PingFang TC", 13, QFont.Bold)
        p.setFont(f)
        p.drawText(QRectF(r.left() + 14, r.top() + 8, r.width() - 50, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setPen(QColor(255, 255, 255, 215) if on else QColor(MUTED))
        p.setFont(QFont("PingFang TC", 11))
        p.drawText(QRectF(r.left() + 14, r.top() + 28, r.width() - 50, 18), Qt.AlignLeft | Qt.AlignVCenter, self.sub)
        c = QPointF(r.right() - 22, r.center().y())
        if on:
            p.setBrush(QColor("white"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(c, 7, 7)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#9AA5BB"), 1.4))
            p.drawEllipse(c, 7, 7)


class MatrixCell(QAbstractButton):
    """Cell of the measurement matrix: done (check), selected (gradient) or pending."""

    def __init__(self):
        super().__init__()
        self.done = False
        self.selected = False
        self.current = False
        self.disabled_cell = False
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(34)
        self.setMinimumWidth(60)

    def set_state(self, done, selected=False, current=False, disabled=False):
        self.done, self.selected, self.current, self.disabled_cell = done, selected, current, disabled
        self.setEnabled(not disabled)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        check = self.done
        if self.disabled_cell:
            p.setBrush(QColor("#F6F8FC"))
            p.setPen(QPen(QColor("#EDF0F7"), 1))
            p.drawRoundedRect(r, 10, 10)
            return
        if self.done:                                   # finished: blue, white check
            paint_gradient_rect(p, r.adjusted(1, 1, -1, -1) if self.selected else r, 10)
            col = QColor("white")
        else:                                           # pending: light grey, faint check
            p.setBrush(QColor("#EEF2F9") if not self.underMouse() else QColor("#E6ECF7"))
            p.setPen(QPen(QColor("#E1E7F2"), 1))
            p.drawRoundedRect(r, 10, 10)
            col = QColor("#B4BDCD")
        if self.selected or self.current:               # the cell being viewed / measured: bright ring
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(BRIGHT if self.selected else COBALT), 2.2))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 10, 10)
        p.setPen(QPen(col, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        c = r.center()
        p.drawPolyline([QPointF(c.x() - 6, c.y() + 0.5), QPointF(c.x() - 2, c.y() + 4.5), QPointF(c.x() + 6, c.y() - 4)])


class MatrixView(QWidget):
    """Grid of MatrixCells with row / column headers.  rows: [(key, label)], cols: [(key, label)]."""
    cellClicked = Signal(object, object)

    def __init__(self, rows, cols):
        super().__init__()
        from PySide6.QtWidgets import QGridLayout
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("matrixview")
        self.setStyleSheet("QWidget#matrixview { background: transparent; }")
        self.g = QGridLayout(self)
        self.g.setContentsMargins(0, 0, 0, 0)
        self.g.setHorizontalSpacing(8)
        self.g.setVerticalSpacing(8)
        self.cells = {}
        self.rebuild(rows, cols)

    def rebuild(self, rows, cols):
        while self.g.count():
            it = self.g.takeAt(0)
            wd = it.widget()
            if wd is not None:
                wd.hide()
                wd.setParent(None)
                wd.deleteLater()
        self.cells = {}
        for j, (ck, cl) in enumerate(cols):
            lb = QLabel(cl)
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(f"color:{MUTED};font-size:11px;")
            self.g.addWidget(lb, 0, j + 1)
        for i, (rk, rl) in enumerate(rows):
            lb = QLabel(rl)
            lb.setStyleSheet(f"color:{TEXT};font-size:12px;font-weight:600;")
            self.g.addWidget(lb, i + 1, 0)
            for j, (ck, cl) in enumerate(cols):
                c = MatrixCell()
                c.clicked.connect(lambda _=False, r=rk, c_=ck: self.cellClicked.emit(r, c_))
                self.g.addWidget(c, i + 1, j + 1)
                self.cells[(rk, ck)] = c
        self.g.setColumnStretch(0, 0)
        for j in range(len(cols)):
            self.g.setColumnStretch(j + 1, 1)


class DashedMeter(QWidget):
    """Row of short dashes (blue gradient) like the level meter in the design."""

    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setFixedHeight(10)

    def set_value(self, v):
        self.value = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = 22
        w = self.width() / n
        for i in range(n):
            t = i / (n - 1)
            c = QColor(BRIGHT).lighter(130) if t > self.value else QColor(BRIGHT)
            if t <= self.value:
                c = QColor(int(42 + (20 - 42) * t), int(168 + (86 - 168) * t), int(240 + (209 - 240) * t))
            else:
                c = QColor("#CFE3F6")
            p.setBrush(c)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(i * w + 1.5, 2, w - 3, 5), 2.5, 2.5)


class LegendChip(QWidget):
    def __init__(self, text, color, hollow=False):
        super().__init__()
        self.text, self.color, self.hollow = text, color, hollow
        self.setFixedHeight(30)
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance(text) + 52)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor("#F6F8FC"))
        p.setPen(QPen(QColor("#E1E7F2"), 1))
        p.drawRoundedRect(r, 15, 15)
        c = QPointF(r.left() + 16, r.center().y())
        if self.hollow:
            p.setBrush(QColor("white"))
            p.setPen(QPen(QColor(self.color), 1.6))
        else:
            p.setBrush(QColor(self.color))
            p.setPen(Qt.NoPen)
        p.drawEllipse(c, 4.5, 4.5)
        p.setPen(QColor(TEXT))
        p.setFont(QFont("PingFang TC", 11, QFont.DemiBold))
        p.drawText(QRectF(c.x() + 12, r.top(), r.width() - 30, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text)


class NavButton(QAbstractButton):
    def __init__(self, icon, text):
        super().__init__()
        self.icon_name, self.text_ = icon, text
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(40)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        on = self.isChecked()
        if on:
            paint_gradient_rect(p, r, 11)
        elif self.underMouse():
            p.setBrush(QColor("#E8EEF9"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 11, 11)
        col = "white" if on else MUTED
        p.drawPixmap(14, int((r.height() - 20) / 2), nav_icon(self.icon_name, col))
        p.setPen(QColor(col))
        p.setFont(QFont("PingFang TC", 13, QFont.Bold if on else QFont.DemiBold))
        p.drawText(QRectF(44, 0, r.width() - 50, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text_)


class DeviceCard(QFrame):
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(3)
        a = QLabel("輸出裝置")
        a.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self.name = QLabel("—")
        self.name.setStyleSheet("font-weight:800;font-size:14px;")
        self.name.setWordWrap(True)
        row = QHBoxLayout()
        self.dot = QLabel("●")
        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        row.addWidget(self.dot)
        row.addWidget(self.status, 1)
        v.addWidget(a)
        v.addWidget(self.name)
        v.addLayout(row)
        self.set_status("未啟動", False)

    def set_status(self, text, ok):
        self.dot.setStyleSheet(f"color:{COBALT if ok else '#B4BCCD'};font-size:10px;")
        self.status.setText(text)

    def mousePressEvent(self, e):
        self.clicked.emit()


class MicView(QWidget):
    """Mic-position diagram: speakers on top, mics at the bottom, gradient path of the active measurement."""

    micClicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(260)
        self.mics = []          # [(pos_key, label)] left→right
        self.active_pos = None
        self.active_kind = "L"
        self._rects = {}

    def set_mics(self, mics):
        self.mics = mics
        self.update()

    def set_active(self, pos, kind):
        self.active_pos, self.active_kind = pos, kind
        self.update()

    def mousePressEvent(self, e):
        for k, r in self._rects.items():
            if r.contains(QPointF(e.position())):
                self.micClicked.emit(k)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        box = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        p.setBrush(QColor("#F8FAFE"))
        pen = QPen(QColor("#CBD3E4"), 1.2)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(box, 16, 16)
        sp_y = box.top() + 36
        spk = {"L": QPointF(box.left() + box.width() * 0.26, sp_y), "R": QPointF(box.left() + box.width() * 0.74, sp_y)}
        n = len(self.mics)
        my = box.bottom() - 62
        mic_pts = {}
        for i, (k, lab) in enumerate(self.mics):
            x = box.left() + box.width() * (0.5 + (i - (n - 1) / 2) * 0.15)
            mic_pts[k] = QPointF(x, my)
        kind = self.active_kind
        if self.active_pos in mic_pts:
            targets = [spk["L"]] if kind == "L" else [spk["R"]] if kind == "R" else [spk["L"], spk["R"]]
            if kind.startswith("S"):
                targets = []
            for t in targets:
                g = QLinearGradient(t, mic_pts[self.active_pos])
                g.setColorAt(0, QColor("#7CB4F0"))
                g.setColorAt(1, QColor(COBALT))
                p.setPen(QPen(QBrush(g), 2.4))
                p.drawLine(QPointF(t.x(), t.y() + 16), mic_pts[self.active_pos])
        for name, pt in (("左喇叭", spk["L"]), ("右喇叭", spk["R"])):
            r = QRectF(pt.x() - 34, pt.y() - 15, 68, 30)
            p.setBrush(QColor(NAVY))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor("white"))
            p.setFont(QFont("PingFang TC", 11, QFont.Bold))
            p.drawText(r, Qt.AlignCenter, name)
        self._rects = {}
        for k, lab in self.mics:
            c = mic_pts[k]
            on = k == self.active_pos
            rr = QRectF(c.x() - 17, c.y() - 17, 34, 34)
            self._rects[k] = rr
            if on:
                p.setBrush(QColor("white"))
                p.setPen(QPen(QColor(COBALT), 2))
                p.drawEllipse(rr.adjusted(-4, -4, 4, 4))
                paint_gradient_rect(p, rr, 17, deep=True, glow=False)
                ic = nav_icon("mic", "white", 18)
            else:
                p.setBrush(QColor("white"))
                p.setPen(QPen(QColor("#D6DDEC"), 1.4))
                p.drawEllipse(rr)
                ic = nav_icon("mic", MUTED, 18)
            p.drawPixmap(int(c.x() - 9), int(c.y() - 9), ic)
            p.setPen(QColor(TEXT))
            p.setFont(QFont("PingFang TC", 10, QFont.DemiBold))
            p.drawText(QRectF(c.x() - 40, c.y() + 22, 80, 16), Qt.AlignCenter, lab)
