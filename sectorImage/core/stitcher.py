from . import singleImage
import numpy as np
import multiprocessing as mp
import pickle as pickle
from .toolkit import pickler
import matplotlib.pyplot as plt


class Stitcher:

    def __init__(self, fns, mpflag=True):
        self.fns = fns
        self.images = []
        self.mpflag = mpflag

        print('Preprocessing images')

    def loadImages(self):

        if self.mpflag:
            ncpus = mp.cpu_count()
            pool = mp.Pool(ncpus)
            self.images = pool.map(singleRoutine, self.fns)
            pool.close()
            pool.join()
        else:
            for fn in self.fns:
                npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
                im = singleImage.SingleImage(fn)
                im.getFeatures(npz=npzfn)
                im.setFeatures(npz=npzfn)
                im.getLines()
                self.images.append(im)

    def stitchImages(self, plot=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, image in enumerate(self.images):
            for rs, phis in zip(image.walker.rsplines, image.walker.phis):
                phis = np.array(phis) + i * 1900
                ax.plot(phis, rs, color='black', lw=0.5)

        fig.savefig('img/out/stitched.png', dpi=300)

    def pickleSave(self, fn='stitcher.pkl'):
        p = pickler.Pickler()
        p.save(self, fn)


def singleRoutine(fn):
    npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
    im = singleImage.SingleImage(fn)
    im.getFeatures(npz=npzfn)
    im.setFeatures(npz=npzfn)
    im.getLines()
    return im
