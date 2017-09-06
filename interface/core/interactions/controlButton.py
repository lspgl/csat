from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..tools.stylesheet import StyleSheet
from ..tools.hoverListener import mouseoverEvent


class ControlButton(QLabel):
    # Large button to control functions

    def __init__(self, parent, pos, color, name, function=None, fargs=None, hover=True):
        # Inherit from label
        QLabel.__init__(self, parent)
        self.parent = parent
        self.color = color
        self.name = name
        self.pos = pos

        self.function = function
        if fargs is not None:
            self.fargs = fargs
        else:
            self.fargs = []

        # Enable hover
        if hover:
            self.hoverReady()

        # Dimensions
        self.setGeometry(QRect(0, 0, 150, 30))

        # Initialize Stylesheet
        s = ('border: 5px;' +
             'background-color: ' + self.color['main'] + ';' +
             'color:' + self.color['text'] + ';' +
             'border-radius:5px;' +
             'font-family: "Roboto";' +
             'font-weight: 500;' +
             'font-size: 12px;')
        self.style = StyleSheet(s)
        self.setStyleSheet(self.style.get())

        # Set button text
        self.setText(self.name.upper())
        self.setAlignment(Qt.AlignCenter)

        self.setCursor(Qt.PointingHandCursor)

        # Move to position
        self.move(*self.pos)

    def hoverReady(self):
        # Add event listener
        self.filter = mouseoverEvent(self)
        self.installEventFilter(self.filter)

    def infunc(self):
        # Hover-in func
        self.style.set(background_color=self.color['hover'], color=self.color['hoverText'], font_weight=700)
        self.setStyleSheet(self.style.get())

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QColor('#000000'))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

    def outfunc(self):
        # Hover-out func
        self.style.set(background_color=self.color['main'], color=self.color['text'], font_weight=500)
        self.setStyleSheet(self.style.get())
        self.setGraphicsEffect(None)

    def mousePressEvent(self, ev):
        x, y = self.pos
        self.move(x, y + 1)

    def mouseReleaseEvent(self, ev):
        self.move(*self.pos)
        if self.function is not None:
            self.function(*self.fargs)
