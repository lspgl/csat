from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..tools.stylesheet import StyleSheet
from ..tools.hoverListener import mouseoverEvent


class SateLabel(QLabel):
    # Label to indicated system readyness state

    def __init__(self, parent, pos, color, hover=False):
        # Inherit from label
        QLabel.__init__(self, parent)

        # Initialize hover
        if hover:
            self.hoverReady()

        self.color = color
        self.setGeometry(QRect(0, 0, 30, 11))

        # Initialize Style
        s = ("border: 5px;" +
             "background-color: " + self.color + ";" +
             "border-radius:5px;")
        self.style = StyleSheet(s)
        self.setStyleSheet(self.style.get())
        self.move(*pos)

    def updateText(self, text):
        # Change containing text
        self.setText(text)

    def blink(self, c1, c2):
        # Blink label color
        def tick():
            # Tick function called at each timeout event
            if self.color != c1:
                self.color = c1
            else:
                self.color = c2
            self.style.set(background_color=self.color)
            self.setStyleSheet(self.style.get())
        # Start Qt Timer
        self.labelTimer = QTimer()
        self.labelTimer.timeout.connect(tick)
        self.labelTimer.start(500)

    def steady(self, c='#00ce4e'):
        # Set steady color of label
        if hasattr(self, 'labelTimer'):
            self.labelTimer.timeout.disconnect()
        self.style.set(background_color=c)
        self.setStyleSheet(self.style.get())

    def hoverReady(self):
        self.filter = mouseoverEvent(self)
        self.installEventFilter(self.filter)

    def infunc(self):
        pass

    def outfunc(self):
        pass
