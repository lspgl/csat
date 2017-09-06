from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ...tools.stylesheet import StyleSheet

from .mplCanvas import MplCanvas


class ElectrodePanel(QLabel):

    def __init__(self, parent, vAnchor=0):
        QLabel.__init__(self, parent)
        self.setGeometry(QRect(0, vAnchor, parent.width(), parent.height() / 2.0))
        self.margin = 50
        self.setContentsMargins(*[self.margin] * 4)
        self.setAlignment(Qt.AlignTop)
        # self.setText('This will be an electrodePanel')
        # self.setStyleSheet('background-color: #800000')

        self.statField = Statistics(self)
        self.plotField = Plot(self)


class Statistics(QLabel):

    def __init__(self, parent, width=400):
        QLabel.__init__(self, parent)
        self.setGeometry(QRect(0, 0, width, parent.height()))
        # self.setStyleSheet('background-color: #000080')

    def pushStats(self, data):
        # Number of data fields
        numfields = len(data)
        # Individual field height
        fieldHeight = self.height() / float(numfields)
        # Right margin of key field
        margin = [0, 0, 20, 0]

        self.keyfields = {}
        self.valuefields = {}
        for i, key in enumerate(data):
            # Vertical Anchor Point
            vAnchor = i * fieldHeight
            # Current key field
            ckf = QLabel(self)
            ckf.setGeometry(QRect(0, vAnchor, self.width() / 2, fieldHeight))
            ckf.setContentsMargins(*margin)
            ckf.setStyleSheet("qproperty-alignment: 'AlignVCenter | AlignRight'; qproperty-wordWrap: true;")

            # Content is the data key
            ckf.setText(str(key).upper())
            self.keyfields[key] = ckf

            # Current value field
            cvf = QLabel(self)
            cvf.setGeometry(QRect(self.width() / 2.0, vAnchor, self.width() / 2.0, fieldHeight))
            cvf.setStyleSheet("font-weight: 700; font-size:30px;")

            # Content is the data value
            cvf.setText(str(data[key]))
            self.valuefields[key] = cvf


class Plot(QLabel):

    def __init__(self, parent):
        QLabel.__init__(self, parent)

        self.setGeometry(QRect(parent.statField.width(), 0, parent.width() - parent.statField.width(), parent.height()))
        self.canvas = MplCanvas(self)
