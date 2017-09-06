from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..tools.stylesheet import StyleSheet
from .subSpaces import electrodePanel


class Space(QLabel):

    def __init__(self, controller):
        # Inherit from label
        QLabel.__init__(self, controller.parent)
        self.setGeometry(controller.geometry)
        self.setContentsMargins(100, 100, 100, 100)
        self.setAlignment(Qt.AlignTop)
        s = ("color:" + controller.parent.textgray + ';'
             )
        self.style = StyleSheet(s)
        self.setStyleSheet(self.style.get())

        data = {'mu': 10.0, 'std': 50.0, 'ovality': 1e-3}

        self.tPanel = electrodePanel.ElectrodePanel(self)
        self.tPanel.statField.pushStats(data)

        self.bPanel = electrodePanel.ElectrodePanel(self, vAnchor=self.height() / 2.0)
        self.bPanel.statField.pushStats(data)
