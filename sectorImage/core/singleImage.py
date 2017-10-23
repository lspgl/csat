from . import image
from . import linewalker
from . import lineParser as lp
from . import midpoints
import numpy as np


class SingleImage:

    def __init__(self, fn):
        """
        Class for handling a single image

        Parameters
        ----------

        fn: string
            filename of source image
        """
        self.fn = fn

    def getFeatures(self, radius=5000, interpolationOrder=1, resolution=None, npz=None):
        """
        Process the features of the image

        Parameters
        ----------
        radius: float or int, optional
            maximal radius which is analyzed. Depending on the midpoint this determines the maximally covered angle of the detectable region. Default is 5000
        interpolationOrder: int, optional
            order of the interpolation polynom for the coordinate transform. Default is 1
        resolution: int, optional
            number of pixel in an interpolation line. If None is given, the radius is taken as the size of a measurement line. Default is None
        npz: bool, optional
            If True the results of the feature detection is stored in an .npz file. Default is None

        """
        self.img = image.Image(self.fn)

        angularLines, self.angles = self.img.lineSweep(radius,
                                                       resolution=resolution,
                                                       interpolationOrder=interpolationOrder)
        self.coverage = self.img.thetaCovered
        # self.features, self.loss = self.img.detectFeatures(angularLines, plot=True)
        self.features = self.img.detectFeatures(angularLines, plot=False)

        # self.angles = self.angles[int(self.loss / 2):int(-self.loss / 2)]
        if npz is not None:
            print('Storing Features in', npz)
            np.savez(npz, f=self.features, a=self.angles)

    def setFeatures(self, npz):
        """
        Recover features from an npz file

        Parameters
        ----------
        npz: string
            filename of the npz datafile
        """
        print('Recovering Features from', npz)
        npzfile = np.load(npz)
        self.features = npzfile['f']
        self.angles = npzfile['a']

    def getLines(self):
        """
        Parametrize the lines of the binary image

        Returns
        -------
        (r,phi): ndarray
            parametrized coordinates of the band midpoints
        """
        self.walker = midpoints.Walker(self.features)
        self.r, self.phi = self.walker.walkSkeleton(plot=False, maxwidth=10,)
        #self.walker = linewalker.Walker()
        # self.walker.scanMultiple(self.features)
        #self.rbands, self.phis = self.walker.fastFeatures(plot=True)
        return self.r, self.phi

    def __repr__(self):
        retstr = 'Image Processing Object for ' + str(self.fn)
        return retstr
