from . import image
from . import linewalker
from . import lineParser as lp
from . import midpoints
import numpy as np
import time
import matplotlib.pyplot as plt


class SingleImage:

    def __init__(self, fn, calibration):
        """
        Class for handling a single image

        Parameters
        ----------

        fn: string
            filename of source image
        """
        self.fn = fn
        self.calibration = calibration

    def getFeatures(self, resolution=None, npz=None, lock=None):
        """
        Process the features of the image

        Parameters
        ----------
        radius: float or int, optional
            maximal radius which is analyzed. Depending on the midpoint this determines the maximally covered angle of the detectable region. Default is 5000
        resolution: int, optional
            number of pixel in an interpolation line. If None is given, the radius is taken as the size of a measurement line. Default is None
        npz: bool, optional
            If True the results of the feature detection is stored in an .npz file. Default is None

        """
        self.img = image.Image(self.fn, self.calibration, lock=lock)
        angularLines, self.angles, self.radii = self.img.transformRadial(plot=False)
        self.features, self.start = self.img.detectFeatures(angularLines, plot=False)

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
        """fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.walker.skeleton)
        ax.set_aspect('auto')
        figfn = self.fn.split('.')[0].split('/')[-1] + '.png'
        fig.savefig(figfn, dpi=1200)"""
        self.r, self.phi = self.walker.walkSkeleton(plot=False, maxwidth=10,)
        return self.r, self.phi

    def __repr__(self):
        retstr = 'Image Processing Object for ' + str(self.fn)
        return retstr
