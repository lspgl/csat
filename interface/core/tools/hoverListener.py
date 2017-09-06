from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class mouseoverEvent(QObject):

    def __init__(self, parent):
        super(mouseoverEvent, self).__init__(parent)
        self.parent = parent

    def eventFilter(self, object, event):
        if event.type() == QEvent.Enter:
            self.parent.infunc()
            return True
        elif event.type() == QEvent.Leave:
            self.parent.outfunc()
            return True
        return False
