import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ...tools import colors

import numpy as np

cols = colors.Colors()


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent, dpi=300):

        font = {
            'family': 'Roboto',
            'size': 3
        }
        matplotlib.rc('font', **font)

        w_inch = parent.width() / dpi
        h_inch = parent.height() / dpi
        self.fig = Figure(figsize=(w_inch, h_inch), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)

        self.fig.patch.set_facecolor((0, 0, 0, 0))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor((0, 0, 0, 0))

        self.fig.subplots_adjust(bottom=0.15)

        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')

        self.ax.yaxis.label.set_color(cols.textgray)
        self.ax.xaxis.label.set_color(cols.textgray)
        for key in self.ax.spines:
            self.ax.spines[key].set_color(cols.textgray)

        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_linewidth(0.2)
        self.ax.spines['left'].set_linewidth(0.2)

        self.ax.grid(True, which='both', ls='--', lw=0.2)
        self.ax.tick_params(axis='both', colors=cols.textgray)

        self.build_data()

        self.fig.savefig('test.png', transparent=True)
        self.setParent(parent)

    def build_data(self):
        y = np.random.rand(1000)
        x = np.linspace(0, 1, 1000)
        self.ax.plot(x, y, lw=0.4, color=cols.highlight)
