"""roomcorr_widgets.py - interactive frequency-response / target-curve editor."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

FMIN, FMAX = 20.0, 20000.0
COARSE = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
NICE = (10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000)
FLAT = [[20.0, 0.0], [100.0, 0.0], [1000.0, 0.0], [10000.0, 0.0], [20000.0, 0.0]]


def paint_tip(p, rc, x, y, lines):
    """Small dark read-out box next to the cursor.  lines: str or (color, str)."""
    p.setFont(QFont("PingFang TC", 11))
    fm = p.fontMetrics()
    items = [(None, ln) if isinstance(ln, str) else ln for ln in lines]
    w = max(fm.horizontalAdvance(t) for _, t in items) + 24
    h = fm.height() * len(items) + 12
    bx = x + 14 if x + 14 + w < rc.right() else x - 14 - w
    by = min(max(y - h / 2, rc.top()), rc.bottom() - h)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(27, 42, 74, 232))
    p.drawRoundedRect(QRectF(bx, by, w, h), 8, 8)
    for i, (c, t) in enumerate(items):
        ty = by + 6 + i * fm.height()
        if c:
            p.setBrush(QColor(c))
            p.drawEllipse(QRectF(bx + 8, ty + fm.height() / 2 - 3, 6, 6))
        p.setPen(QColor("white"))
        p.drawText(QRectF(bx + 18 if c else bx + 10, ty, w, fm.height()), Qt.AlignLeft | Qt.AlignVCenter, t)
        p.setPen(Qt.NoPen)


class CurveWidget(QWidget):
    """Shows measured/predicted curves (dB relative to mid-band) and an editable target curve.
    Left-drag a point to move it, double-click empty space to add one, right-click a point to delete it."""
    changing = Signal()
    released = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self.series = []                      # (label, color, fx, y, width, dashed)
        self.points = [list(p) for p in FLAT]
        self.ymin, self.ymax = -30.0, 15.0
        self._drag = None
        self._tip = ""
        self.m = (46, 10, 26, 32)
        self.editable = True
        self.legend = True
        self.show_target = True
        self.xmin, self.xmax = FMIN, FMAX
        self.ystep = 5
        self.yunit = "dB"
        self._hover = None

    # ---- data
    def set_xrange(self, lo, hi):
        self.xmin, self.xmax = float(lo), float(hi)
        self.update()

    def set_range(self, ymin, ymax, step=5):
        self.ymin, self.ymax, self.ystep = ymin, ymax, step
        self.update()

    def set_series(self, series):
        self.series = series
        self.update()

    def set_points(self, pts):
        self.points = [[float(f), float(d)] for f, d in sorted(pts)]
        self.update()

    def get_points(self):
        return [(f, d) for f, d in sorted(self.points)]

    # ---- geometry
    def rect(self):
        l, t, r, b = self.m
        return QRectF(l, t, self.width() - l - r, self.height() - t - b)

    def fx(self, f):
        rc = self.rect()
        return rc.left() + (math.log10(f) - math.log10(self.xmin)) / (math.log10(self.xmax) - math.log10(self.xmin)) * rc.width()

    def fy(self, db):
        rc = self.rect()
        return rc.bottom() - (db - self.ymin) / (self.ymax - self.ymin) * rc.height()

    def x2f(self, x):
        rc = self.rect()
        u = min(max((x - rc.left()) / rc.width(), 0), 1)
        return 10 ** (math.log10(self.xmin) + u * (math.log10(self.xmax) - math.log10(self.xmin)))

    def y2db(self, y):
        rc = self.rect()
        return self.ymin + (rc.bottom() - y) / rc.height() * (self.ymax - self.ymin)

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rc = self.rect()
        p.fillRect(self.rect().adjusted(-60, -20, 30, 40), QColor("#FFFFFF"))
        p.setFont(QFont("Helvetica", 9))
        wide = self.xmax / self.xmin > 60
        ticks = COARSE if wide else NICE
        for f in [t for t in ticks if self.xmin <= t <= self.xmax]:
            x = self.fx(f)
            p.setPen(QPen(QColor("#E8EDF6"), 1))
            p.drawLine(int(x), int(rc.top()), int(x), int(rc.bottom()))
            p.setPen(QColor("#55617A"))
            p.drawText(int(x) - 18, int(rc.bottom()) + 16, 36, 14, Qt.AlignCenter, f"{f / 1000:g}k" if f >= 1000 else f"{f:g}")
        for db in range(int(np.ceil(self.ymin / self.ystep) * self.ystep), int(self.ymax) + 1, int(self.ystep)):
            y = self.fy(db)
            p.setPen(QPen(QColor("#C5CEE0" if db == 0 else "#EDF1F8"), 1.2 if db == 0 else 1))
            p.drawLine(int(rc.left()), int(y), int(rc.right()), int(y))
            p.setPen(QColor("#55617A"))
            p.drawText(2, int(y) - 7, 40, 14, Qt.AlignRight | Qt.AlignVCenter, f"{db:+d}" if db else "0")
        p.setClipRect(rc)
        for label, color, fx, y, w, dashed in self.series:
            path = QPainterPath()
            for i, (a, b) in enumerate(zip(fx, y)):
                pt = (self.fx(min(max(a, self.xmin), self.xmax)), self.fy(float(b)))
                path.moveTo(*pt) if i == 0 else path.lineTo(*pt)
            pen = QPen(QColor(color), w)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        if not self.editable:
            p.setClipping(False)
            x = rc.left() + 8
            fm = p.fontMetrics()
            for label, color, *_ in (self.series if self.legend else []):
                p.setPen(QColor(color))
                p.drawText(int(x), int(rc.top()) + 14, label)
                x += fm.horizontalAdvance(label) + 14
            self._paint_hover(p)
            p.end()
            return
        # target
        pts = sorted(self.points)
        fg = np.geomspace(FMIN, FMAX, 300)
        tg = np.interp(np.log2(fg), np.log2([q[0] for q in pts]), [q[1] for q in pts])
        path = QPainterPath()
        for i, (a, b) in enumerate(zip(fg, tg)):
            path.moveTo(self.fx(a), self.fy(b)) if i == 0 else path.lineTo(self.fx(a), self.fy(b))
        p.setPen(QPen(QColor("#33415C"), 2))
        p.drawPath(path)
        p.setClipping(False)
        for i, (f, d) in enumerate(self.points):
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(QColor("#E8820C" if i == self._drag else "#1456D1"))
            p.drawEllipse(self.fx(f) - 6, self.fy(d) - 6, 12, 12)
        # legend
        x = rc.left() + 8
        fm = p.fontMetrics()
        for label, color, *_ in (self.series if self.legend else []):
            p.setPen(QColor(color))
            p.drawText(int(x), int(rc.top()) + 14, label)
            x += fm.horizontalAdvance(label) + 14
        p.setPen(QColor("#33415C"))
        p.drawText(int(x), int(rc.top()) + 14, "目標（可拖曳）")
        if self._tip:
            p.setPen(QColor("#E8820C"))
            p.drawText(int(rc.right()) - 170, int(rc.top()) + 30, 165, 14, Qt.AlignRight, self._tip)
        if self._drag is None:
            self._paint_hover(p)
        p.end()

    # ---- hover read-out (every chart shows the data under the mouse)
    def hover_lines(self, f):
        lines = [f"{f:,.0f} Hz" if f >= 100 else f"{f:.1f} Hz"]
        lf = math.log10(f)
        for label, color, fx, y, *_ in self.series:
            if not label or len(fx) < 2:
                continue
            fx = np.asarray(fx, float)
            if f < fx[0] or f > fx[-1]:
                continue
            v = float(np.interp(lf, np.log10(np.maximum(fx, 1e-9)), np.asarray(y, float)))
            lines.append((color, f"{label}　{v:+.1f} {self.yunit}" if self.yunit == "dB" else f"{label}　{v:.2f} {self.yunit}"))
        if self.editable and self.show_target and self.points:
            pts = sorted(self.points)
            v = float(np.interp(math.log2(f), np.log2([q[0] for q in pts]), [q[1] for q in pts]))
            lines.append(("#33415C", f"目標　{v:+.1f} dB"))
        return lines

    def _paint_hover(self, p):
        if self._hover is None:
            return
        rc = self.rect()
        x, y = self._hover.x(), self._hover.y()
        if not rc.contains(self._hover):
            return
        f = self.x2f(x)
        p.setClipping(False)
        p.setPen(QPen(QColor(27, 42, 74, 110), 1, Qt.DashLine))
        p.drawLine(int(x), int(rc.top()), int(x), int(rc.bottom()))
        p.drawLine(int(rc.left()), int(y), int(rc.right()), int(y))
        lines = self.hover_lines(f)
        lines.append(("#8A94A8", f"游標　{self.y2db(y):+.1f} {self.yunit}" if self.yunit == "dB" else f"游標　{self.y2db(y):.2f} {self.yunit}"))
        paint_tip(p, rc, x, y, lines)

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    # ---- editing
    def _nearest(self, pos, rad=12):
        best, bd = None, rad
        for i, (f, d) in enumerate(self.points):
            dist = math.hypot(self.fx(f) - pos.x(), self.fy(d) - pos.y())
            if dist < bd:
                best, bd = i, dist
        return best

    def mousePressEvent(self, e):
        if not self.editable:
            return
        i = self._nearest(e.position())
        if e.button() == Qt.LeftButton:
            self._drag = i
        elif e.button() == Qt.RightButton and i is not None and len(self.points) > 2:
            del self.points[i]
            self.update()
            self.changing.emit()
            self.released.emit()

    def mouseDoubleClickEvent(self, e):
        if not self.editable:
            return
        if e.button() == Qt.LeftButton and self._nearest(e.position()) is None and self.rect().contains(e.position()):
            f = self.x2f(e.position().x())
            d = round(min(max(self.y2db(e.position().y()), -15), 15), 1)
            self.points.append([f, d])
            self.points.sort()
            self.update()
            self.changing.emit()
            self.released.emit()

    def mouseMoveEvent(self, e):
        self._hover = e.position()
        if not self.editable or self._drag is None:
            self.update()
            return
        i = self._drag
        f = self.x2f(e.position().x())
        lo = self.points[i - 1][0] * 1.08 if i > 0 else FMIN
        hi = self.points[i + 1][0] / 1.08 if i < len(self.points) - 1 else FMAX
        f = min(max(f, lo), max(hi, lo))
        d = round(min(max(self.y2db(e.position().y()), -15), 15) * 2) / 2
        self.points[i] = [f, d]
        self._tip = f"{f:.0f} Hz  {d:+.1f} dB"
        self.update()
        self.changing.emit()

    def mouseReleaseEvent(self, e):
        if not self.editable:
            return
        if self._drag is not None:
            self._drag = None
            self._tip = ""
            self.update()
            self.released.emit()


# ====================================================================== time-frequency charts
def _cmap(u):
    """REW-style spectrogram colours: purple -> blue -> cyan -> green -> yellow -> red (loudest), u in [0, 1]."""
    stops = np.array([[0.00, 40, 10, 70], [0.18, 30, 40, 200], [0.36, 20, 160, 230], [0.52, 40, 200, 90],
                      [0.70, 220, 230, 40], [0.86, 255, 140, 20], [1.00, 230, 20, 20]], float)
    u = np.clip(u, 0, 1)
    return np.stack([np.interp(u, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], -1).astype(np.uint8)


def _decay_color(u):
    """Decay-time strip: white (short) -> deep red (long), u in [0, 1]."""
    u = float(np.clip(u, 0, 1))
    return QColor(int(255 - 30 * u), int(245 - 215 * u), int(240 - 220 * u))


class _TFBase(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.m = (52, 12, 70, 34)
        self.xmin, self.xmax = 20.0, 20000.0
        self._hover = None
        self.empty_text = "沒有資料"

    def rect(self):
        l, t, r, b = self.m
        return QRectF(l, t, self.width() - l - r, self.height() - t - b)

    def set_xrange(self, lo, hi):
        self.xmin, self.xmax = float(lo), float(hi)
        self.update()

    def fx(self, f, rc=None):
        rc = rc or self.rect()
        return rc.left() + (math.log10(f) - math.log10(self.xmin)) / (math.log10(self.xmax) - math.log10(self.xmin)) * rc.width()

    def x2f(self, x, rc=None):
        rc = rc or self.rect()
        u = min(max((x - rc.left()) / rc.width(), 0), 1)
        return 10 ** (math.log10(self.xmin) + u * (math.log10(self.xmax) - math.log10(self.xmin)))

    def _freq_axis(self, p, rc):
        p.setFont(QFont("Helvetica", 9))
        ticks = COARSE if self.xmax / self.xmin > 60 else NICE
        for f in [t for t in ticks if self.xmin <= t <= self.xmax]:
            x = self.fx(f, rc)
            p.setPen(QColor("#55617A"))
            p.drawText(int(x) - 18, int(rc.bottom()) + 16, 36, 14, Qt.AlignCenter, f"{f / 1000:g}k" if f >= 1000 else f"{f:g}")

    def mouseMoveEvent(self, e):
        self._hover = e.position()
        self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()


class SpectrogramWidget(_TFBase):
    """Frequency (log, x) vs time (y, down = later) heat map of an impulse response."""

    def __init__(self):
        super().__init__()
        self.t = self.f = self.Z = None
        self.decay = None
        self.dyn = 50.0
        self._img = None
        self.m = (52, 26, 70, 34)

    def set_data(self, t_ms, f, Z, dyn=None):
        self.t, self.f, self.Z = (None, None, None) if Z is None else (np.asarray(t_ms), np.asarray(f), np.asarray(Z))
        if dyn:
            self.dyn = float(dyn)
        self._img = None
        self.decay = None
        if Z is not None:
            Z = self.Z
            k0 = np.argmax(Z, axis=0)
            dec = np.full(Z.shape[1], self.t[-1] - self.t[0])
            for j in range(Z.shape[1]):
                k = np.where((np.arange(len(self.t)) > k0[j]) & (Z[:, j] <= Z[k0[j], j] - 20))[0]
                if len(k):
                    dec[j] = self.t[k[0]] - self.t[k0[j]]
            self.decay = dec
        self.update()

    def set_dyn(self, dyn):
        self.dyn = float(dyn)
        self._img = None
        self.update()

    def _image(self, w, h):
        lf = np.log10(self.f)
        xs = np.linspace(math.log10(self.xmin), math.log10(self.xmax), max(w, 2))
        ys = np.linspace(self.t[0], self.t[-1], max(h, 2))
        ji = np.clip(np.interp(xs, lf, np.arange(len(lf))), 0, len(lf) - 1)
        ti = np.clip(np.interp(ys, self.t, np.arange(len(self.t))), 0, len(self.t) - 1)
        Zi = self.Z[np.round(ti).astype(int)][:, np.round(ji).astype(int)]
        Zi[:, (xs < lf[0]) | (xs > lf[-1])] = -999
        rgb = _cmap((Zi + self.dyn) / self.dyn)
        a = np.ascontiguousarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))
        img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 4, QImage.Format_RGBA8888)
        return img.copy()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.contentsRect(), QColor("#FFFFFF"))
        rc = self.rect()
        if self.Z is None:
            p.setPen(QColor("#8A94A8"))
            p.drawText(rc, Qt.AlignCenter, self.empty_text)
            return
        key = (int(rc.width()), int(rc.height()), self.xmin, self.xmax, self.dyn)
        if self._img is None or self._img[0] != key:
            self._img = (key, self._image(int(rc.width()), int(rc.height())))
        p.drawImage(rc.topLeft(), self._img[1])
        if self.decay is not None:                       # decay-time strip: the longer a frequency rings, the redder
            lim = max(float(np.percentile(self.decay, 95)), 1.0)
            xs = np.array([self.fx(v, rc) for v in self.f])
            sy = rc.top() - 16
            for j in range(len(self.f) - 1):
                p.fillRect(QRectF(xs[j], sy, max(xs[j + 1] - xs[j], 1), 10), _decay_color(self.decay[j] / lim))
            p.setPen(QColor("#55617A"))
            p.setFont(QFont("Helvetica", 9))
            p.drawText(2, int(sy) - 2, 60, 12, Qt.AlignLeft, "消散")
        self._freq_axis(p, rc)
        p.setPen(QColor("#55617A"))
        span = self.t[-1] - self.t[0]
        step = 50 if span > 300 else (20 if span > 120 else 10)
        for tm in np.arange(np.ceil(self.t[0] / step) * step, self.t[-1] + 1e-9, step):
            y = rc.top() + (tm - self.t[0]) / span * rc.height()
            p.drawText(2, int(y) - 7, 44, 14, Qt.AlignRight | Qt.AlignVCenter, f"{tm:.0f}")
        p.drawText(2, int(rc.top()) - 2, 60, 12, Qt.AlignLeft, "ms")
        # colour bar
        cb = QRectF(rc.right() + 16, rc.top(), 12, rc.height())
        for i in range(int(cb.height())):
            c = _cmap(np.array([1 - i / cb.height()]))[0]
            p.setPen(QColor(int(c[0]), int(c[1]), int(c[2])))
            p.drawLine(int(cb.left()), int(cb.top() + i), int(cb.right()), int(cb.top() + i))
        p.setPen(QColor("#55617A"))
        for k in range(0, int(self.dyn) + 1, 10):
            y = cb.top() + k / self.dyn * cb.height()
            p.drawText(int(cb.right()) + 3, int(y) - 7, 36, 14, Qt.AlignLeft | Qt.AlignVCenter, f"-{k}")
        if self._hover is not None and rc.contains(self._hover):
            x, y = self._hover.x(), self._hover.y()
            f = self.x2f(x, rc)
            tm = self.t[0] + (y - rc.top()) / rc.height() * span
            j = int(np.argmin(np.abs(np.log10(self.f) - math.log10(f))))
            i = int(np.argmin(np.abs(self.t - tm)))
            lines = [f"{f:,.0f} Hz　{tm:.0f} ms", ("#2AA8F0", f"強度　{self.Z[i, j]:.1f} dB")]
            if self.decay is not None:
                lines.append(("#E8382A", f"消散（降 20 dB）　{self.decay[j]:.0f} ms"))
            p.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.DashLine))
            p.drawLine(int(x), int(rc.top()), int(x), int(rc.bottom()))
            p.drawLine(int(rc.left()), int(y), int(rc.right()), int(y))
            paint_tip(p, rc, x, y, lines)
        p.end()


def _rainbow(u):
    """REW waterfall surface colours by level: navy -> blue -> cyan -> green -> yellow -> orange -> red."""
    stops = np.array([[0.00, 10, 20, 90], [0.15, 20, 60, 200], [0.32, 20, 170, 230], [0.48, 40, 190, 60],
                      [0.66, 200, 230, 30], [0.82, 250, 160, 20], [1.00, 230, 20, 20]], float)
    u = np.clip(u, 0, 1)
    return np.stack([np.interp(u, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], -1)


class WaterfallWidget(_TFBase):
    """REW-style 3-D cumulative spectral decay surface, software-rasterised per pixel (Gouraud level colours,
    lighting, z-buffer, slice outlines).  Right-drag rotates, wheel zooms, double-click resets."""

    HOME = (-32.0, 28.0, 1.0)

    def __init__(self):
        super().__init__()
        self.t = self.f = self.Z = None
        self.floor = -45.0
        self.m = (12, 12, 12, 12)
        self.yaw, self.pitch, self.zoom = self.HOME
        self._rot = None
        self._fast = False
        self._cache = None
        self._pick = None
        self.setMinimumHeight(380)

    def set_data(self, t_ms, f, Z, floor=None):
        self.t, self.f, self.Z = (None, None, None) if Z is None else (np.asarray(t_ms), np.asarray(f), np.asarray(Z))
        if floor:
            self.floor = float(floor)
        self._cache = None
        self.update()

    def set_xrange(self, lo, hi):
        super().set_xrange(lo, hi)
        self._cache = None

    # ---- projection (orthographic; x = log f, y = time into the screen, z = level)
    def _proj(self, u, v, z):
        rc = self.rect()
        ya, pa = np.radians(self.yaw), np.radians(self.pitch)
        x = (np.asarray(u, float) - 0.5) * 1.0
        y = (np.asarray(v, float) - 0.5) * 0.6                 # t = 0 at the back, later slices come forward (CSD convention)
        xr = x * np.cos(ya) - y * np.sin(ya)
        yr = x * np.sin(ya) + y * np.cos(ya)
        zz = (np.asarray(z, float) - 0.5) * 0.52
        sx = xr
        sy = -zz * np.cos(pa) + yr * np.sin(pa)
        depth = -(yr * np.cos(pa) + zz * np.sin(pa))            # larger = farther from the viewer
        s = min(rc.width(), rc.height() * 1.6) * 0.74 * self.zoom
        return rc.center().x() + sx * s, rc.center().y() - 6 + sy * s * 0.9, depth

    def _grid(self):
        sel = (self.f >= self.xmin) & (self.f <= self.xmax)
        f = self.f[sel]
        Z = np.clip(self.Z[:, sel], self.floor, 0.0)
        u = (np.log10(f) - np.log10(self.xmin)) / (np.log10(self.xmax) - np.log10(self.xmin))
        v = (self.t - self.t[0]) / max(self.t[-1] - self.t[0], 1e-9)
        z = (Z - self.floor) / (0.0 - self.floor)
        return f, u, v, z

    def _time_ticks(self):
        span = self.t[-1] - self.t[0]
        step = 50.0 if span > 220 else 25.0 if span > 120 else 10.0
        ts = np.arange(np.ceil(self.t[0] / step) * step, self.t[-1] + 1e-6, step)
        return [(float((tt - self.t[0]) / span), f"{tt:.0f}") for tt in ts]

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.contentsRect(), QColor("#FFFFFF"))
        rc = self.rect()
        if self.Z is None:
            p.setPen(QColor("#8A94A8"))
            p.drawText(rc, Qt.AlignCenter, self.empty_text)
            return
        key = (int(rc.width()), int(rc.height()), self.yaw, self.pitch, self.zoom, self.xmin, self.xmax, self._fast)
        if self._cache is None or self._cache[0] != key:
            img = QImage(self.size(), QImage.Format_ARGB32_Premultiplied)
            img.fill(QColor("#FFFFFF"))
            q = QPainter(img)
            q.setRenderHint(QPainter.Antialiasing, True)
            q.setRenderHint(QPainter.TextAntialiasing, True)
            self._draw_scene(q)
            q.end()
            self._cache = (key, img)
        p.drawImage(0, 0, self._cache[1])
        if self._hover is not None and self._rot is None:
            self._draw_hover(p)
        p.end()

    def _box(self, p, back_u, back_v, fu, ft, tt, dz, dbs, front=False):
        """Axes box: back walls and floor grid (front=False) or the front outline + labels (front=True)."""
        P = lambda u, v, z: QPointF(*self._proj(u, v, z)[:2])
        if not front:
            p.setPen(QPen(QColor("#DCE2EC"), 1))
            for uu in fu:
                p.drawLine(P(uu, 0, 0), P(uu, 1, 0)); p.drawLine(P(uu, back_v, 0), P(uu, back_v, 1))
            for vv, _ in tt:
                p.drawLine(P(0, vv, 0), P(1, vv, 0)); p.drawLine(P(back_u, vv, 0), P(back_u, vv, 1))
            for zz in dz:
                p.drawLine(P(0, back_v, zz), P(1, back_v, zz)); p.drawLine(P(back_u, 0, zz), P(back_u, 1, zz))
            p.setPen(QPen(QColor("#B9C2D0"), 1.2))
            for (a, b) in (((0, back_v, 0), (1, back_v, 0)), ((back_u, 0, 0), (back_u, 1, 0)),
                           ((0, back_v, 1), (1, back_v, 1)), ((back_u, 0, 1), (back_u, 1, 1)),
                           ((back_u, back_v, 0), (back_u, back_v, 1)), ((1 - back_u, back_v, 0), (1 - back_u, back_v, 1)),
                           ((back_u, 1 - back_v, 0), (back_u, 1 - back_v, 1))):
                p.drawLine(P(*a), P(*b))
            return
        fv, fu_ = 1.0 - back_v, 1.0 - back_u
        p.setPen(QPen(QColor("#9AA5B8"), 1.2))
        p.drawLine(P(0, fv, 0), P(1, fv, 0)); p.drawLine(P(fu_, 0, 0), P(fu_, 1, 0))
        p.drawLine(P(fu_, fv, 0), P(fu_, fv, 1))
        p.setFont(QFont("Helvetica", 9))
        p.setPen(QColor("#55617A"))
        fmt = lambda t: f"{t / 1000:g}k" if t >= 1000 else f"{t:g}"
        # frequency: front floor edge + back top edge
        for uu, t in zip(fu, ft):
            x, y, _ = self._proj(uu, fv, 0)
            p.drawText(int(x) - 22, int(y) + 4, 44, 14, Qt.AlignCenter, fmt(t))
            x, y, _ = self._proj(uu, back_v, 1)
            p.drawText(int(x) - 22, int(y) - 17, 44, 14, Qt.AlignCenter, fmt(t))
        # time: front-side floor edge + far-side top edge
        for vv, s in tt:
            x, y, _ = self._proj(fu_, vv, 0)
            p.drawText(int(x) + (5 if fu_ > 0.5 else -47), int(y) - 2, 42, 14, Qt.AlignLeft if fu_ > 0.5 else Qt.AlignRight, s + ("ms" if s == tt[-1][1] else "m"))
            x, y, _ = self._proj(back_u, vv, 1)
            p.drawText(int(x) + (5 if back_u > 0.5 else -47), int(y) - 30, 42, 14, Qt.AlignLeft if back_u > 0.5 else Qt.AlignRight, s + "m")
        # level: left-most and right-most vertical edges
        corners = [(uu, vv) for uu in (0.0, 1.0) for vv in (0.0, 1.0)]
        xs = [self._proj(uu, vv, 0.5)[0] for uu, vv in corners]
        for (uu, vv), left in ((corners[int(np.argmin(xs))], True), (corners[int(np.argmax(xs))], False)):
            for zz, d in zip(dz, dbs):
                x, y, _ = self._proj(uu, vv, zz)
                p.drawText(int(x) + (-46 if left else 6), int(y) - 7, 40, 14, Qt.AlignRight if left else Qt.AlignLeft, f"{d:d}" + (" dB" if d == dbs[0] else ""))
        p.setFont(QFont("Helvetica", 10))
        x, y, _ = self._proj(0.5, back_v, 1)
        p.drawText(int(x) - 50, int(y) - 36, 100, 14, Qt.AlignCenter, "Frequency (Hz)")
        x, y, _ = self._proj(back_u, 0.5, 1)
        p.drawText(int(x) - 50, int(y) - 66, 100, 14, Qt.AlignCenter, "Time")
        p.setFont(QFont("Helvetica", 9))
        p.setPen(QColor("#8A94A8"))
        p.drawText(self.rect().adjusted(0, 0, -6, -4), Qt.AlignRight | Qt.AlignBottom, "右鍵拖曳旋轉 · 滾輪縮放 · 雙擊重設")

    def _draw_scene(self, p):
        f, u, v, z = self._grid()
        if self._fast:
            su, sv = max(len(u) // 140, 1), max(len(v) // 32, 1)
            u, f, z, v = u[::su], f[::su], z[::sv, ::su], v[::sv]
        back_v = 0.0 if self._proj(0.5, 0, 0.5)[2] > self._proj(0.5, 1, 0.5)[2] else 1.0
        back_u = 0.0 if self._proj(0, 0.5, 0.5)[2] > self._proj(1, 0.5, 0.5)[2] else 1.0
        ticks = COARSE if self.xmax / self.xmin > 60 else NICE
        ft = [t for t in ticks if self.xmin <= t <= self.xmax]
        fu = [(np.log10(t) - np.log10(self.xmin)) / (np.log10(self.xmax) - np.log10(self.xmin)) for t in ft]
        dbs = list(range(0, int(self.floor) - 1, -10))
        dz = [(d - self.floor) / (0.0 - self.floor) for d in dbs]
        tt = self._time_ticks()
        self._box(p, back_u, back_v, fu, ft, tt, dz, dbs)
        w, h = int(self.width()), int(self.height())
        rgba = self._render(u, v, z, w, h, back_u, back_v)
        img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        p.drawImage(0, 0, img)
        self._box(p, back_u, back_v, fu, ft, tt, dz, dbs, front=True)

    # ---- software rasteriser
    def _render(self, u, v, z, w, h, back_u, back_v):
        nt, nf = z.shape
        U, V = np.meshgrid(u, v)
        X, Y, D = self._proj(U, V, z)
        # vertices: surface, then curtains (front slice, both frequency ends) dropping to the floor
        X0, Y0, D0 = self._proj(U, V, np.zeros_like(z))
        px = [X.ravel(), Y.ravel(), D.ravel(), z.ravel(), np.ones(nt * nf)]
        tris = []
        idx = np.arange(nt * nf).reshape(nt, nf)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        tris.append(np.stack([a, b, c], 1)); tris.append(np.stack([a, c, d], 1))
        base = nt * nf
        curt = []
        fv_i = 0 if back_v == 1.0 else nt - 1
        for sel_top, sel_bot, dark in ((idx[fv_i, :], None, 0.55), (idx[:, 0], None, 0.5), (idx[:, -1], None, 0.5)):
            top = sel_top
            n = len(top)
            start = base + sum(len(cv) for cv in curt)
            curt.append(top)
            px[0] = np.concatenate([px[0], X0.ravel()[top]]); px[1] = np.concatenate([px[1], Y0.ravel()[top]])
            px[2] = np.concatenate([px[2], D0.ravel()[top] + 1e-4]); px[3] = np.concatenate([px[3], np.zeros(n)])
            px[4] = np.concatenate([px[4], np.full(n, dark)])
            bot = np.arange(start, start + n)
            tris.append(np.stack([top[:-1], top[1:], bot[1:]], 1)); tris.append(np.stack([top[:-1], bot[1:], bot[:-1]], 1))
        T = np.concatenate(tris).astype(np.int64)
        Xv, Yv, Dv, Lv, Kv = [np.asarray(q, np.float32) for q in px]
        img = np.zeros((h, w, 4), np.uint8)
        zbuf = np.full((h, w), np.inf, np.float32)
        light = np.array([-0.35, -0.55, -0.76], np.float32)
        light /= np.linalg.norm(light)
        # triangle normals in screen space (x, y, depth) -> lighting
        e1 = np.stack([Xv[T[:, 1]] - Xv[T[:, 0]], Yv[T[:, 1]] - Yv[T[:, 0]], (Dv[T[:, 1]] - Dv[T[:, 0]]) * 300], 1)
        e2 = np.stack([Xv[T[:, 2]] - Xv[T[:, 0]], Yv[T[:, 2]] - Yv[T[:, 0]], (Dv[T[:, 2]] - Dv[T[:, 0]]) * 300], 1)
        nrm = np.cross(e1, e2)
        nn = np.linalg.norm(nrm, axis=1) + 1e-9
        lam = np.abs(nrm @ light) / nn
        shade = (0.62 + 0.5 * lam).astype(np.float32) * Kv[T[:, 0]]
        for c0 in range(0, len(T), 12000):
            self._raster(T[c0:c0 + 12000], shade[c0:c0 + 12000], Xv, Yv, Dv, Lv, img, zbuf, w, h)
        if not self._fast:
            self._slice_lines(X, Y, D, img, zbuf, w, h)
        return np.ascontiguousarray(img)

    def _raster(self, T, shade, Xv, Yv, Dv, Lv, img, zbuf, w, h):
        x0, x1, x2 = Xv[T[:, 0]], Xv[T[:, 1]], Xv[T[:, 2]]
        y0, y1, y2 = Yv[T[:, 0]], Yv[T[:, 1]], Yv[T[:, 2]]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        keep = np.abs(area) > 1e-6
        T, shade, area = T[keep], shade[keep], area[keep]
        x0, x1, x2, y0, y1, y2 = x0[keep], x1[keep], x2[keep], y0[keep], y1[keep], y2[keep]
        if not len(T):
            return
        bx0 = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)), 0, w - 1).astype(np.int32)
        bx1 = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)), 0, w - 1).astype(np.int32)
        by0 = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)), 0, h - 1).astype(np.int32)
        by1 = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)), 0, h - 1).astype(np.int32)
        bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
        npx = (bw * bh).astype(np.int64)
        tot = int(npx.sum())
        if tot == 0:
            return
        tri = np.repeat(np.arange(len(T)), npx)
        off = np.cumsum(npx) - npx
        k = np.arange(tot) - off[tri]
        px = bx0[tri] + k % bw[tri]
        py = by0[tri] + k // bw[tri]
        cx, cy = px + 0.5, py + 0.5
        inv = 1.0 / area[tri]
        w0 = ((x1[tri] - cx) * (y2[tri] - cy) - (x2[tri] - cx) * (y1[tri] - cy)) * inv
        w1 = ((x2[tri] - cx) * (y0[tri] - cy) - (x0[tri] - cx) * (y2[tri] - cy)) * inv
        w2 = 1.0 - w0 - w1
        eps = -0.002
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        tri, px, py, w0, w1, w2 = tri[inside], px[inside], py[inside], w0[inside], w1[inside], w2[inside]
        if not len(tri):
            return
        Tt = T[tri]
        depth = (w0 * Dv[Tt[:, 0]] + w1 * Dv[Tt[:, 1]] + w2 * Dv[Tt[:, 2]]).astype(np.float32)
        level = w0 * Lv[Tt[:, 0]] + w1 * Lv[Tt[:, 1]] + w2 * Lv[Tt[:, 2]]
        order = np.argsort(-depth, kind="stable")                 # far -> near; last write wins
        tri, px, py, depth, level = tri[order], px[order], py[order], depth[order], level[order]
        zc = np.full((h, w), np.inf, np.float32)
        zc[py, px] = depth
        vis = (depth <= zc[py, px]) & (depth < zbuf[py, px])
        px, py, depth, level, tri = px[vis], py[vis], depth[vis], level[vis], tri[vis]
        col = _rainbow(level) * shade[tri][:, None]
        img[py, px, :3] = np.clip(col, 0, 255).astype(np.uint8)
        img[py, px, 3] = 255
        zbuf[py, px] = depth

    def _slice_lines(self, X, Y, D, img, zbuf, w, h):
        """Thin dark outline along every time slice, depth-tested against the surface."""
        nt, nf = X.shape
        xa, xb = X[:, :-1].ravel(), X[:, 1:].ravel()
        ya, yb = Y[:, :-1].ravel(), Y[:, 1:].ravel()
        da, db = D[:, :-1].ravel(), D[:, 1:].ravel()
        n = (np.ceil(np.maximum(np.abs(xb - xa), np.abs(yb - ya))) + 1).astype(np.int64)
        seg = np.repeat(np.arange(len(xa)), n)
        off = np.cumsum(n) - n
        s = (np.arange(int(n.sum())) - off[seg]) / np.maximum(n[seg] - 1, 1)
        px = np.clip(np.round(xa[seg] + (xb[seg] - xa[seg]) * s), 0, w - 1).astype(np.int32)
        py = np.clip(np.round(ya[seg] + (yb[seg] - ya[seg]) * s), 0, h - 1).astype(np.int32)
        dd = da[seg] + (db[seg] - da[seg]) * s
        vis = dd <= zbuf[py, px] + 0.004
        px, py = px[vis], py[vis]
        img[py, px, :3] = (img[py, px, :3].astype(np.float32) * 0.62).astype(np.uint8)

    def _draw_hover(self, p):
        f, u, v, z = self._grid()
        U, V = np.meshgrid(u, v)
        X, Y, _ = self._proj(U, V, z)
        hx, hy = self._hover.x(), self._hover.y()
        d2 = (X - hx) ** 2 + (Y - hy) ** 2
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if d2[i, j] > 30 ** 2:
            return
        sel = (self.f >= self.xmin) & (self.f <= self.xmax)
        Zs = self.Z[:, sel]
        col = Zs[:, j]
        k20 = np.where(col <= col[0] - 20)[0]
        p.setPen(QPen(QColor("#1B2A4A"), 2))
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(X[i, j], Y[i, j]), 4, 4)
        lines = [f"{f[j]:,.1f} Hz　{self.t[i]:.1f} ms", ("#E8820C", f"{Zs[i, j]:+.1f} dB（相對 0 ms 最強處）"),
                 ("#E8382A", f"降 20 dB　{self.t[k20[0]]:.0f} ms" if len(k20) else f"降 20 dB　> {self.t[-1]:.0f} ms")]
        paint_tip(p, self.rect(), hx, hy, lines)

    # ---- interaction
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._rot = (e.position(), self.yaw, self.pitch)
            self._fast = True
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._rot is not None:
            p0, y0, p0_ = self._rot
            d = e.position() - p0
            self.yaw = y0 + d.x() * 0.4
            self.pitch = float(np.clip(p0_ + d.y() * 0.3, -10, 80))
            self._cache = None
            self.update()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rot is not None and e.button() == Qt.RightButton:
            self._rot = None
            self._fast = False
            self._cache = None
            self.unsetCursor()
            self.update()

    def mouseDoubleClickEvent(self, e):
        self.yaw, self.pitch, self.zoom = self.HOME
        self._cache = None
        self.update()

    def wheelEvent(self, e):
        self.zoom = float(np.clip(self.zoom * (1.1 if e.angleDelta().y() > 0 else 1 / 1.1), 0.5, 3.0))
        self._cache = None
        self.update()
