from . import singleImage
import numpy as np
import multiprocessing as mp
import pickle as pickle
from .toolkit import pickler
from .toolkit.parmap import Parmap
import matplotlib.pyplot as plt
import sys
import os


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

    def loadImages(self):
        """
        Initialize the SingleImage instances and process the images.
        """

        if self.mpflag:
            # mp.set_start_method('spawn')
            #ncpus = mp.cpu_count()
            #pool = mp.Pool(ncpus)
            self.images = Parmap(self.singleRoutine, self.fns)

            # pool.close()
            # pool.join()
        else:
            for fn in self.fns:
                # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
                im = singleImage.SingleImage(fn)
                # im.getFeatures(npz=npzfn)
                im.getFeatures()
                # im.setFeatures(npz=npzfn)
                im.getLines()
                self.images.append(im)

    def stitchImages(self, plot=False):
        """
        Stitch the parametrized band midpoints and plot the output

        Parameters
        ----------
        plot: bool, optional
            plot the output. Default True
        """
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ref_point = 0
            for i, image in enumerate(self.images):
                for rs, phis in zip(image.r, image.phi):
                    #ref_point += len(phis) * image.coverage
                    #phis = np.array(phys) + ref_point
                    phis = np.array(phis) + i * 190
                    ax.plot(phis, rs, color='black', lw=0.5)

            fig.savefig('img/out/stitched.png', dpi=300)

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

    def singleRoutine(self, fn):
        """
        Image processing routine to be parallelized

        Parameters
        ----------
        fn: string
            filename of the single image
        """
        # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
        im = singleImage.SingleImage(fn)
        # im.getFeatures(npz=npzfn)
        im.getFeatures()
        # im.setFeatures(npz=npzfn)
        im.getLines()
        return im
