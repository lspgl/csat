import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# from core import core, core
from core import interactions, spaces
from core.tools import colors


class GUI(QMainWindow):

    def __init__(self, app):
        print('Initializing GUI')
        super(GUI, self).__init__()
        self.app = app

        c = colors.Colors().__dict__
        for key in c:
            setattr(self, key, c[key])

        self.initUI()

    def initUI(self):
        # QFontDatabase.addApplicationFont("/resources/fonts/Roboto-Regular.ttf")
        # QFontDatabase.addApplicationFont("/resources/fonts/Roboto-Italic.ttf")
        # QFontDatabase.addApplicationFont("/resources/fonts/Roboto-Light.ttf")
        # QFontDatabase.addApplicationFont("/resources/fonts/Roboto-Lightitalic.ttf")

        self.setWindowTitle('CSAT')
        self.setStyleSheet("background-color:" + self.darkgray + ";")

        self.setAutoFillBackground(True)
        self.screen = self.app.desktop().screenGeometry()

        self.mainField()
        self.windowButtons()
        self.sideBar()

        self.line.setFocus()

        self.app.desktop().primaryScreen()
        self.showFullScreen()

        self.move(self.app.desktop().screenGeometry(1).topLeft())

    def mousePressEvent(self, event):
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, interactions.TextInput):
            focused_widget.clearFocus()
        QMainWindow.mousePressEvent(self, event)

    def paintEvent(self, e):

        qp = QPainter()
        qp.begin(self)

        pos = (0, 0)
        w = self.screen.width()
        h = 30
        color = self.lightgray
        self.drawRectangles(qp, color, pos, w, h)

        pos = (0, 31)
        w = 200
        h = self.screen.height() - 30
        color = self.midgray
        self.drawRectangles(qp, color, pos, w, h)

        qp.end()

    def drawRectangles(self, qp, c, pos, w, h):
        color = QColor(c)
        qp.setPen(color)
        qp.setBrush(color)
        x, y = pos
        qp.drawRect(x, y, w, h)

    def windowButtons(self):
        self.exitBtn = interactions.WindowButton(self, (25, 10), self.controlred)
        self.exitBtn.mouseReleaseEvent = self.exitApp

        self.state = interactions.SateLabel(self, (45, 10), self.controlgreen, hover=False)
        # self.state.blink(self.controlred, self.controlyel)
        # self.state.steady(self.controlgreen)

    def sideBar(self):
        self.line = interactions.TextInput(self)
        self.line.setPlaceholderText('Serial')
        self.line.move(25, 100)
        self.line.setFixedWidth(150)
        self.line.setNumericString()

        colors = {'main': self.highlight, 'hover': self.highlight, 'text': '#000000', 'hoverText': self.darkgray}
        self.runBtn = interactions.ControlButton(
            self, (25, 150), colors, 'Run', function=self.contentController.loadSpace, fargs=['core.spaces.runLabel'])

        colors = {'main': self.highlight, 'hover': self.highlight, 'text': '#000000', 'hoverText': self.darkgray}
        self.configureBtn = interactions.ControlButton(
            self, (25, 210), colors, 'Configure', function=self.contentController.loadSpace, fargs=['core.spaces.configurationLabel'])

        colors = {'main': self.darkgray, 'hover': self.controlyel, 'text': '#848484', 'hoverText': self.darkgray}
        self.calibrateBtn = interactions.ControlButton(self, (25, self.screen.height() - 60), colors, 'Calibrate')

    def mainField(self):
        spaceGeometry = QRect(201, 31, self.screen.width() - 200, self.screen.height() - 30)
        self.contentController = spaces.Controller(self, spaceGeometry)
        self.contentController.loadSpace('core.spaces.splashLabel')
        # self.contentController.loadSpace('core.spaces.configurationLabel')

    def exitApp(self, ev=None):
        print('Closing GUI')
        self.app.quit()


def main():
    app = QApplication(sys.argv)
    w = GUI(app)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
