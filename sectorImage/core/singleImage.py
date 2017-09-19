from . import image
from . import linewalker
from . import lineParser as lp
import numpy as np


class SingleImage:

    def __init__(self, fn):
        self.fn = fn

    def getFeatures(self, radius=5000, interpolationOrder=1, resolution=None, npz=None):
        self.img = image.Image(self.fn)
        angularLines, self.angles = self.img.lineSweep(radius,
                                                       resolution=resolution,
                                                       interpolationOrder=interpolationOrder)
        # self.features, self.loss = self.img.detectFeatures(angularLines, plot=True)
        self.features = self.img.detectFeatures(angularLines, plot=False)
        # self.angles = self.angles[int(self.loss / 2):int(-self.loss / 2)]
        if npz is not None:
            print('Storing Features in', npz)
            np.savez(npz, f=self.features, a=self.angles)

    def setFeatures(self, npz):
        print('Recovering Features from', npz)
        npzfile = np.load(npz)
        self.features = npzfile['f']
        self.angles = npzfile['a']

    def getLines(self):
        """
        self.walker = linewalker.Walker()
        self.walker.scanMultiple(self.features)
        self.rbands, self.phis = self.walker.fastFeatures(plot=True)
        """
        self.rbands, self.phis = lp.LineParser(self.features)

    def __repr__(self):
        retstr = 'Image Processing Object for ' + str(self.fn)
        return retstr
