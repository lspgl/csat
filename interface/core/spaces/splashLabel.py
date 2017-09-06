from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Space(QLabel):

    def __init__(self, controller):
        # Inherit from label
        QLabel.__init__(self, controller.parent)
        self.setGeometry(controller.geometry)

        imageFrame = QLabel(self)
        image = QPixmap('resources/img/csatlogo.png')
        imageFrame.setPixmap(image)

        imageFrame.move((self.width() - image.width()) / 2,
                        (self.height() - image.height()) / 2)
