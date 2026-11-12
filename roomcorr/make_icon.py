# generates icon_1024.png (used by the installer)
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QPen, QLinearGradient
from PySide6.QtCore import Qt, QRectF
app = QGuiApplication(sys.argv)
S = 1024
img = QImage(S, S, QImage.Format_ARGB32)
img.fill(Qt.transparent)
p = QPainter(img)
p.setRenderHint(QPainter.Antialiasing)
g = QLinearGradient(0, 0, 0, S)
g.setColorAt(0, QColor("#2b2f36")); g.setColorAt(1, QColor("#15171a"))
p.setBrush(g); p.setPen(Qt.NoPen)
p.drawRoundedRect(QRectF(60, 60, S - 120, S - 120), 200, 200)
pen = QPen(QColor("#3fb950")); pen.setWidth(46); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
import math
n = 9
for i in range(n):
    x = 230 + i * (S - 460) / (n - 1)
    h = 90 + 300 * abs(math.sin(i * 0.9 + 0.4)) * (1 - abs(i - n / 2) / (n * 0.8))
    p.drawLine(int(x), int(S / 2 - h), int(x), int(S / 2 + h))
p.end()
img.save("icon_1024.png")
