import numpy as np
import scipy
from scipy import ndimage
import scipy.fftpack as fft
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from .toolkit import vectools
import math


class Calibrator:

    def __init__(self, fns):
        self.fns = fns
        print('Reading images', fns)
        self.images = [ndimage.imread(fn, flatten=True) for fn in self.fns]

    def sweepFFT(self, i, image):
        # Generate an interpolated profile of the image matrix between points
        # p0, p1 with a resolution defined by the distance between the points
        # to avoid unnecessary interpolation where no information exists

        print('running image', i)
        height, width = image.shape
        xs = np.linspace(0, height, height)
        lp = [(x, 0) for x in xs]
        ep = [(x, width) for x in xs[::-1]]
        pairs = [[l, e] for l, e in zip(lp, ep)]

        zs = [self.interpolate(image, *p) for p in pairs]

        """
        zs = ndimage.median_filter(zs, size=100)

        smoothing = 1e8
        x = np.linspace(0, len(zs), len(zs))
        spline = scipy.interpolate.UnivariateSpline(x, zs, s=smoothing)
        zs_spline = spline(x)
        max_point = np.argmax(zs_spline)

        ri = pairs[max_point][1]
        li = pairs[max_point][0]

        delta = ri[0] - li[0]
        theta = math.atan2(delta, 6000)
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(image)
        """
        ax1.plot([li[1], ri[1]], [li[0], ri[0]])
        ax2.axvline(x=max_point)
        ax2.plot(zs_spline)
        ax2.plot(zs)
        """
        # print(zs)
        ax2.imshow(zs, aspect='auto')

        fig.savefig('img/out/calibration_' + str(i) + '.png', dpi=300)
        theta = 0
        return theta

    def interpolate(self, img, p0, p1, interpolationOrder=1):
        x0, y0 = p0
        x1, y1 = p1
        res = vectools.pointdist(p0, p1)
        res = int(res)
        res = 6000
        x, y = np.linspace(x0, x1, res), np.linspace(y0, y1, res)

        zi = ndimage.map_coordinates(img, np.vstack((x, y)), order=interpolationOrder, mode='nearest')
        zi_filter = np.abs(ndimage.laplace(ndimage.maximum_filter1d(zi, size=10)))
        zifft = fft.fftshift(fft.fft(zi_filter))
        mz = np.max(np.abs(zifft[len(zifft) // 2 + 10:]))
        # return zi_filter
        # return mz
        return np.ones(len(zifft)) * mz
        return np.abs(zifft)

    def sweepAll(self):
        thetas = [self.sweepFFT(i, im) for i, im in enumerate(self.images)]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(thetas)
        fig.savefig('img/out/thetas.png', dpi=300)