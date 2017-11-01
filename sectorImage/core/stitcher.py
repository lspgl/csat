from . import singleImage
import numpy as np
from .toolkit import pickler
from .toolkit.parmap import Parmap
import matplotlib.pyplot as plt
import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Stitcher:

    def __init__(self, fns, mpflag=True):
        """
        Stitching class to combine multiple processed images

        Parameters
        ----------
        fns: List of strings
            filenames to be read and combined. The combination is done in the order of the supplied list
        mpflag: bool, optional
            multiprocessing flag. If on, the image processing is distributed over the available cores.
            This disables plotting of individual images. Default is True
        """
        self.fns = fns
        self.images = []
        self.mpflag = mpflag

        #self.id = int(self.fn.split('cpt')[-1].split('.')[0])
        calibration_path = __location__ + '/../data/calibration.npy'
        self.calibration = np.load(calibration_path)
        #self.midpoint = calibration[self.id - 1][:-1]

    def loadImages(self):
        """
        Initialize the SingleImage instances and process the images.
        """

        if self.mpflag:
            # mp.set_start_method('spawn')
            #ncpus = mp.cpu_count()
            #pool = mp.Pool(ncpus)
            self.images = Parmap(self.singleRoutine, self.fns, self.calibration)
            # pool.close()
            # pool.join()
        else:
            for fn in self.fns:
                # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
                im = singleImage.SingleImage(fn, self.calibration)
                # im.getFeatures(npz=npzfn)
                im.getFeatures()
                # im.setFeatures(npz=npzfn)
                im.getLines()
                self.images.append(im)

    def stitchImages(self, plot=False):
        print('Stitching images')
        """
        Stitch the parametrized band midpoints and plot the output

        Parameters
        ----------
        plot: bool, optional
            plot the output. Default True
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, image in enumerate(self.images[::-1]):
            stepsize_angles = (image.angles[-1] - image.angles[0]) / len(image.angles)
            stepsize_radii = (image.radii[-1] - image.radii[0]) / len(image.radii)
            for rs, phis in zip(image.r, image.phi):
                phis = (np.array(phis) * stepsize_angles) + image.angles[0] + (i * 2 * np.pi / len(self.fns))
                rs = (np.array(rs) * stepsize_radii)
                ax.plot(phis, rs, color='black', lw=0.5)

        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('Radius [px]')
        fig.savefig(__location__ + '/../img/out/stitched.png', dpi=300)

    def pickleSave(self, fn='stitcher.pkl'):
        """
        Save the class with the processed images to a pickled binary object

        Parameters
        ----------
        fn: string
            filename of the output
        """
        p = pickler.Pickler()
        p.save(self, fn)

    def singleRoutine(self, fn, calibration):
        """
        Image processing routine to be parallelized

        Parameters
        ----------
        fn: string
            filename of the single image
        """
        # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
        im = singleImage.SingleImage(fn, calibration)
        # im.getFeatures(npz=npzfn)
        im.getFeatures()
        # im.setFeatures(npz=npzfn)
        im.getLines()
        return im
