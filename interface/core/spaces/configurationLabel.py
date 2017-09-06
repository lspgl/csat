from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..tools.stylesheet import StyleSheet


class Space(QLabel):

    def __init__(self, controller):
        # Inherit from label
        QLabel.__init__(self, controller.parent)
        self.setGeometry(controller.geometry)
        self.setText('Hello')
        self.setContentsMargins(100, 100, 100, 100)
        self.setAlignment(Qt.AlignTop)
        s = ("color:" + controller.parent.textgray + ';'
             )
        self.style = StyleSheet(s)
        self.setStyleSheet(self.style.get())
        self.btn = QPushButton('hello', self)
        self.btn.clicked.connect(self.callingFunction)

    def callingFunction(self):
        self.setText('Potato')
