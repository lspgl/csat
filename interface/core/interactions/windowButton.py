from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..tools.stylesheet import StyleSheet
from ..tools.hoverListener import mouseoverEvent


class WindowButton(QLabel):
    # Circular window button (close, minimize...)

    def __init__(self, parent, pos, color, hover=True):
        # parent: QT widget
        # pos: position of button as tuple
        # color: main color of the button
        # hover: enable hover functions

        # Inherit from label
        QLabel.__init__(self, parent)
        self.color = color

        # Enable hover
        if hover:
            self.hoverReady()

        # Dimensions
        self.setGeometry(QRect(0, 0, 11, 11))

        # Initialize Stylesheet
        s = ("border: 5px;" +
             "background-color: " + self.color + ";" +
             "border-radius:5px;")
        self.style = StyleSheet(s)
        self.setStyleSheet(self.style.get())

        # Move to position
        self.move(*pos)

        # Set hover image
        self.pixmap = QPixmap('resources/img/cross.png')

    def hoverReady(self):
        # Add event listener
        self.filter = mouseoverEvent(self)
        self.installEventFilter(self.filter)

    def infunc(self):
        # Hover-in func
        self.style.set(background_color='#bc4846')
        self.setStyleSheet(self.style.get())
        self.setPixmap(self.pixmap)

    def outfunc(self):
        # Hover-out func
        self.style.set(background_color=self.color)
        self.setStyleSheet(self.style.get())
        self.clear()
