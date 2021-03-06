import numpy as np
from numpy.linalg import eig, inv
import cv2
import time
import math
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import odr
from scipy import ndimage

import multiprocessing as mp
from .toolkit.parmap import Parmap
from .toolkit.colors import Colors as _C
from .toolkit.ellipse import LSqEllipse

import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Calibrator:

    def __init__(self, fns, mpflag=True):
        """
        Calibration class

        Parameters
        ----------
        fns: list ofstring
            filenames of the images to be calibrated
        """

        self.fns = fns
        self.mpflag = mpflag

    def computeOutline(self, fn, plot=False, smoothing=True, subsampling=False, lock=None):
        """
        Filter single source image to obtain a well defined edge of the calibration cirlce

        Parameters
        ----------
        fn: string
            image name to be processed
        plot: bool, optional
            plot the image and the detected edge points. Default is False
        smoothing: bool, optional
            smooth out the edge points to avoid errors created by dust on the calibration piece. Default is True
        subsampling: bool, optional
            only use a subset of the edge points. Default is False
        lock: mp.Lock() object
            will be used in multiprocessing routine

        Returns
        -------
        data: 2D array
            x and y coordinate arrays for the individual edge points
        """
        # t0 = time.time()
        fn_npy = fn.split('.')[0] + '.npy'
        print(_C.LIGHT + 'Calibrating image ' + _C.BOLD + fn + _C.ENDC)
        if lock is not None:
            with lock:
                src = np.load(fn_npy)
        else:
            src = np.load(fn_npy)

        # src = np.rot90(src)
        # print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        src = src.astype(np.uint8, copy=False)
        # print('Blurring')
        # im = np.empty(np.shape(src), np.uint8)

        inversionMapping = True
        if inversionMapping:
            # c_estimate = (src.shape[1], src.shape[0] // 2)
            dx = 650
            c_estimate = (src.shape[1] + dx, src.shape[0] // 2)
            im = cv2.linearPolar(src, c_estimate, src.shape[1] + dx, cv2.WARP_FILL_OUTLIERS)
        else:
            im = np.copy(src)

        # Gaussian Blur to remove fast features
        kernelsize = 5
        sigma = 1.0
        cv2.GaussianBlur(src=im, ksize=(kernelsize, kernelsize), dst=im, sigmaX=sigma, sigmaY=sigma)

        # print('Convolving')
        # Convolving with kernel
        kheight = 3
        prewitt_kernel_x = np.array([[-1, 0, 1]] * kheight)

        im = im.astype(np.int16, copy=False)
        if not inversionMapping:
            im_y = np.empty(np.shape(src), np.int16)
            cv2.filter2D(src=im, kernel=np.transpose(prewitt_kernel_x), dst=im_y, ddepth=-1)

        cv2.filter2D(src=im, kernel=prewitt_kernel_x, dst=im, ddepth=-1)

        if not inversionMapping:
            scaling = np.max(im) / np.max(im_y)
            im += (np.abs(im_y) * scaling).astype(np.int16, copy=False)
            del im_y
        else:
            # Apparently opencv breaks if the transformed image isn't copied first in multiprocessing.
            # Go figure...
            cpy = np.copy(im)
            im = cv2.linearPolar(cpy, c_estimate, im.shape[1] + dx, cv2.WARP_INVERSE_MAP)
            im *= -1
        filtered = np.copy(im)
        # cv2.threshold(src=im, dst=im, thresh=0, maxval=255, type=cv2.THRESH_TOZERO)[1]
        # np.abs(im, out=im)

        # print('Thresholding')
        # mean_val = np.mean(im)
        # print(np.max(im))
        std_val = np.std(im)
        # thresh = mean_val + 3 * std_val
        thresh = 2 * std_val
        thresh = std_val
        # im *= -1

        cv2.threshold(src=im, dst=im, thresh=thresh, maxval=1, type=cv2.THRESH_BINARY)

        # morph_kernel = np.ones((5, 5), np.uint8)

        # morph_kernel_small = np.ones((3, 3), np.uint8)
        #cv2.morphologyEx(src=im, dst=im, op=cv2.MORPH_OPEN, iterations=1, kernel=morph_kernel)
        #cv2.morphologyEx(src=im, dst=im, op=cv2.MORPH_CLOSE, iterations=1, kernel=morph_kernel)

        im = im.astype(np.uint8, copy=False)
        # im = self.skeletonize(im)
        # cv2.morphologyEx(src=im, dst=im, op=cv2.MORPH_CLOSE, iterations=1, kernel=morph_kernel)
        #cv2.morphologyEx(src=skeleton, dst=skeleton, op=cv2.MORPH_OPEN, iterations=1, kernel=morph_kernel_small)

        pt_x = [np.argmax(line > 0) for i, line in enumerate(im)]
        pt_y = np.arange(0, len(pt_x))
        pt_y = np.array([y for i, y in enumerate(pt_y) if pt_x[i] != 0])
        pt_x = np.array([x for x in pt_x if x != 0])

        # blurCorrection = 3
        # pt_x += blurCorrection
        # pt_x, pt_y = self.centralSubset(pt_x, pt_y, 0.5)
        if subsampling:
            subregion_size = 10
            n_regions = 100
            pt_x_sub, centroids = self.subsampling(pt_x, subregion_size, n_regions)
            data = [pt_x_sub, centroids]
        else:
            data = [pt_x, pt_y]
        smoothing = False

        if smoothing:
            sigma = 11
            data = self.smoothing(data, sigma)

        # plot = True
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data[0], data[1], lw=.1, color='red', alpha=0.5)
            ax.imshow(filtered)
            #ax.set_xlim([5700, 6000])
            #ax.set_ylim([1200, 2700])
            ax.set_aspect('auto')
            fig.savefig(__location__ + '/../img/out/calibration_new_' +
                        fn.split('.')[0].split('/')[-1] + '.png', dpi=900)
        del filtered
        del im
        del src
        return data

    def skeletonize(self, img):
        """
        Morphological skeletonization of the binary image
        The image is eroded, dilated and subtracted before being compared with a logical or until one step before
        the 0-image is obtained. The resulting image is the skeleton.

        Parameters
        ----------
        img: 2D array
            Binary image to be skeletonized

        Returns
        -------
        skeleton: 2D array
            skeletonized image
        """
        # t0 = time.time()
        # Initialize arrays to do computation in place
        skeleton = np.zeros(img.shape, np.uint8)
        eroded = np.zeros(img.shape, np.uint8)
        temp = np.zeros(img.shape, np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            cv2.erode(img, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(img, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            img, eroded = eroded, img  # Swap instead of copy

            if cv2.countNonZero(img) == 0:
                # print('Skeletonization in', str(round(time.time() - t0, 2)), 's')
                del eroded
                del temp
                return skeleton

    def centralSubset(self, x, y, size):
        npts = len(x)
        nnew = int(npts * size)
        xnew = x[(npts - nnew) // 2:(npts + nnew) // 2]
        ynew = y[(npts - nnew) // 2:(npts + nnew) // 2]

        return xnew, ynew

    def subsampling(self, x, n, size):
        """
        Subsample set of points

        Parameters
        ----------
        x: float array
            data points to be subsampled
        n: int
            number of new points
        size: int
            size of the pooling region for the subsample

        Returns
        -------
        x_sub: float array
            subsampled x
        centroids: float array
            subsampled y, starting from size // 2
        """
        centroids = np.linspace(size // 2, len(x - size // 2), num=n, endpoint=True)
        x_sub = np.empty(len(centroids))
        for i, c in enumerate(centroids):
            region = x[int(c) - size // 2:int(c) + size // 2]
            x_sub[i] = region.mean()

        return (x_sub, centroids)

    def smoothing(self, data, sigma):
        """
        Smooth out 1D dataset with a gaussian kernel

        Parameters
        ----------
        data: 2D array
            x-y coordinates to be smoothed
        sigma: float
            STD of the smoothing gaussian

        Returns
        -------
        dnew: 2D array
            smoothed x-y coordinates
        """
        x, y = data
        x = ndimage.minimum_filter1d(x, sigma)
        x = ndimage.maximum_filter1d(x, 4 * sigma)
        # filtered = ndimage.gaussian_filter1d(x, sigma, mode='reflect')
        #filtered[:2 * sigma] = x[:2 * sigma]
        #filtered[-2 * sigma:] = x[-2 * sigma:]
        dnew = x, y

        plot = False
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.plot(filtered, lw=0.5)
            #ax.plot(x, lw=0.5)
            # ax.set_xlim([900, 2500])
            fig.savefig('smooting_debug.png', dpi=300)
        return dnew

    def calc_R(self, x, y, xc, yc):
        """
        Calculate radius for a set of points w.r.t a specified point

        Parameters
        ----------
        x: float array
            x coordinates of data points
        y: float array
            y coordinates of data points
        xc: float
            x coordinate of center point
        yc: float
            y coordinate of center point

        Returns
        -------
        r: float array
            calculated radii for all points
        """
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_odr_multivariate(self, beta, x):
        """
        Objective function for multivariate orthogonal distance regression

        Parameters
        ----------
        beta: float array
            Optimization parameters
        x: float array
            x-Data to be fitted

        Returns
        -------
        r: float array
            values of the residuals
        """
        x_flat = [x[i * 4000:(i + 1) * 4000] for i in range(16)]
        # delta = np.array([(d - beta[0][i])**2 + (self.grid - beta[1][i])**2 -
        # beta[2]**2 for i, d in enumerate(x_flat)])
        delta = np.array([np.abs(np.sqrt((d - beta[i * 2])**2 +
                                         (self.grid - beta[i * 2 + 1]) ** 2) -
                                 beta[-1]) for i, d in enumerate(x_flat)])
        delta = np.ndarray.flatten(delta)
        return delta

    def calc_estimate_odr_multivariate(self, data):
        """
        Calculate initial guess for orthogonal distance regression

        Parameters
        ----------
        data: 2D array
            x-y coordinates of all datapoints in all images

        Returns
        -------
        beta0: float array
            1st order estimate on midpoints
        """
        data = [data.x[i * 4000:(i + 1) * 4000] for i in range(16)]
        # data = data.x
        # x_m = np.empty(len(data))
        # y_m = np.empty(len(data))
        R_m = np.empty(len(data))
        beta0 = np.empty(len(data) * 2 + 1)
        for i, x in enumerate(data):
            x_m = np.mean(x)
            y_m = np.mean(self.grid)
            beta0[i * 2] = x_m
            beta0[i * 2 + 1] = y_m
            R_m[i] = self.calc_R(x, self.grid, x_m, y_m).mean()
        R_m_avg = np.mean(R_m)
        beta0[-1] = R_m_avg
        # beta0 = [x_m, y_m, R_m_avg]
        return beta0

    def optimize_odr(self, data):
        """
        Calculate the orthogonal distance regression for a dataset fitted to a circle

        Parameters
        ----------
        data: 2D array
            x-y coordinates of all datapoints in all images

        Returns
        -------
        calibration: float array
            array of midpoint coordinates and respective radii
        """
        self.grid = data[0][1]
        self.grid_stack = np.array(list(self.grid) * len(data))
        x = np.array(data)[:, 0, :].flatten()
        lsc_data = odr.Data(x, y=1)
        lsc_model = odr.Model(fcn=self.f_odr_multivariate,
                              implicit=True,
                              estimate=self.calc_estimate_odr_multivariate)
        lsc_odr = odr.ODR(lsc_data, lsc_model)
        lsc_out = lsc_odr.run()
        lsc_out.pprint()
        opt = lsc_out.beta

        calibration = []
        for i in range(len(data)):
            xc = opt[i * 2]
            yc = opt[i * 2 + 1]
            r = opt[-1]
            package = [xc, yc, r]
            calibration.append(package)
        return calibration

    def f(self, c, x, y, w=1, fixedR=None):
        Ri = self.calc_R(x, y, *c)
        if fixedR is None:
            delta = Ri - Ri.mean()
        else:
            delta = Ri - fixedR
        delta_w = delta * w
        return delta_w

    def f_multivariate(self, c, *args):
        """
        Objective function for leqast squares regression

        Parameters
        ----------
        c: float array
            array of midpoint coordinates. Flattend over all images by alternating x-y coordinate
        args: 2D array
            x-y coordinates of all datapoints in all images

        Returns
        -------
        delta: float array
            values of the residuals
        """
        Rall = np.empty(0)
        for i, d in enumerate(args):
            x, y = d
            xc = c[i * 2]
            yc = c[i * 2 + 1]
            Ri = self.calc_R(x, y, xc, yc)
            Rall = np.append(Rall, Ri)
        delta = Rall - np.mean(Rall)
        return delta

    def leastsq_circle_multivariate(self, data):
        """
        Calculate the least squares regression for a dataset fitted to a circle

        Parameters
        ----------
        data: 2D array
            x-y coordinates of all datapoints in all images

        Returns
        -------
        calibration: float array
            array of midpoint coordinates and respective radii
        """
        center_estimates = np.empty(0)
        for i, d in enumerate(data):
            x, y = d
            x_m = np.mean(x)
            y_m = np.mean(y)
            mp = np.array([x_m, y_m])
            center_estimates = np.append(center_estimates, mp)

        opt = optimize.least_squares(self.f_multivariate, center_estimates, args=data, jac='2-point')
        calibration = []
        Rs = []
        for i in range(len(opt.x) // 2):
            xc = opt.x[i * 2]
            yc = opt.x[i * 2 + 1]
            center = [xc, yc]
            x, y = data[i]
            Ri = self.calc_R(x, y, *center)
            R = Ri.mean()
            # circle = plt.Circle((xc, yc), R, edgecolor='red', facecolor='none')
            calibration_i = [xc, yc, R]
            calibration.append(calibration_i)
            Rs.append(R)

        return calibration

    def leastsq_circle(self, x, y, w=1, fixedR=None):
        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        # center, ier = optimize.leastsq(self.f, center_estimate, args=(x, y, w, fixedR))
        opt = optimize.least_squares(self.f, center_estimate, args=(x, y, w, fixedR), jac='3-point')
        center = opt['x']
        xc, yc = center
        Ri = self.calc_R(x, y, *center)
        R = Ri.mean()
        residu = np.sum((Ri - R)**2)
        return (xc, yc, R, residu)

    def computeAll(self, tofile=False):
        """
        Compute all calibration midpoints

        Parameters
        ----------
        tofile: bool, optional
            save the calibration to file

        Returns
        -------
        calibration: float array
            array of midpoint coordinates and respective radii
        """
        lock = mp.Lock()
        # self.comp = Parmap(self.computeOutline, self.fns, lock=lock)
        edgePoints = Parmap(self.computeOutline, self.fns, lock=lock)
        #edgePoints = [cmp[:-1][0] for cmp in self.comp]
        #images = [cmp[-1] for cmp in self.comp]
        self.calibration = self.leastsq_circle_multivariate(edgePoints)
        if False:
            for i in range(len(self.calibration)):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(edgePoints[i][0], edgePoints[i][1], lw=0.1, color='red')
                circle = plt.Circle((self.calibration[i][0], self.calibration[i][1]),
                                    self.calibration[i][2], facecolor='none', edgecolor='yellow')
                # ax.imshow(images[i])
                ax.add_artist(circle)
                ax.set_xlim([500, 1000])
                ax.set_aspect('auto')
                fig.savefig('skeletonDebug_' + str(i) + '.png', dpi=300)
        if tofile:
            np.save(__location__ + '/../data/calibration.npy', np.array(self.calibration))
        return np.array(self.calibration)

    def loadCalibration(self, fn):
        """
        Load calibration from file

        Parameters
        ----------
        fn: starting
            filename of calibration data

        Returns
        -------
        calibration: float array
            array of midpoint coordinates and respective radii
        """
        self.calibration = np.load(fn)
        return self.calibration

    def oscillationCircle(self):
        """
        Calculate oscillation of the midpoints

        Returns
        -------
        oscillation: float array
            center and radius of the oscillation circle
        newCalib: float array
            array of midpoint coordinates and respective radii fitted to a perfect circle
        """
        mps = [np.array([x[0], x[1]]) for x in self.calibration]
        # ravg = np.mean(np.array([x[2] for x in self.calibration]))
        radii = np.array([x[2] for x in self.calibration])
        x, y, r, res = self.leastsq_circle(*zip(*mps))
        mp = np.array([x, y])
        mps_zero = mps - mp
        mps_radial = np.array([np.array([np.sqrt(pt[0]**2 + pt[1]**2), np.arctan2(pt[1], pt[0])]) for pt in mps_zero])
        thetas = mps_radial[:, 1]
        dt = np.array([(t - thetas[i + 1]) % np.pi for i, t in enumerate(thetas[:-1])])
        self.shiftAngle = np.mean(dt)
        rs = mps_radial[:, 0]
        dr = r - rs
        dr_true = np.argwhere(np.abs(dr) < 1).flatten()
        # mps_true = np.array(mps)[dr_true]
        radii_true = radii[dr_true]
        mps_true_radial = mps_radial[dr_true]
        spacing = 2 * np.pi / len(mps)
        # dt = [t[1] - mps_true_radial[i + 1][1] for i, t in enumerate(mps_true_radial[:-1])]

        support = np.array([n * spacing for n in dr_true])
        # print(support)
        guess = mps_true_radial[0, 1] - support[0]
        opt = optimize.least_squares(self.arrayDistance, guess, args=(mps_true_radial[:, 1], support))['x']
        # print(opt)
        opt_support = support + opt
        opt_support[opt_support > np.pi] -= 2 * np.pi
        opt_support[opt_support < -np.pi] += 2 * np.pi

        complete_support = np.zeros(len(mps))
        refpoint = opt_support[0]
        refindex = dr_true[0]
        zeropoint = refpoint - refindex * spacing
        for i in range(len(mps)):
            complete_support[i] = i * spacing + zeropoint
        complete_support[complete_support > np.pi] -= 2 * np.pi
        complete_support[complete_support < -np.pi] += 2 * np.pi

        true_radius_osc = np.mean(mps_true_radial[:, 0])
        true_radius_cal = np.mean(radii_true)
        support_cart = np.array([np.array([true_radius_osc * np.cos(phi), true_radius_osc * np.sin(phi)])
                                 for phi in complete_support])
        support_cart += mp

        newCalib = []
        for i in range(len(mps)):
            xnew, ynew = support_cart[i]
            newCalib.append([xnew, ynew, true_radius_cal])

        self.newCalib = newCalib

        # circle_fit = plt.Circle((x, y), r, lw=1, facecolor='none', edgecolor='red')

        x0 = mps[0][0]
        y0 = mps[0][1]

        dx = x0 - x
        dy = y0 - y

        theta = math.atan2(dy, dx)

        oscillation = [x, y, r, theta]
        # print(oscillation)
        return oscillation, newCalib

    def arrayDistance(self, shift, data, equal):
        """
        Calculate the radial distance between two arrays

        Parameters
        ----------
        shift: float
            angular shift of the equalized data
        data: float array
            fixed data to be compared
        equal: float array
            equalized data to be compared

        Returns
        -------
        distances: float array
            distances between all points
        """
        shifted = equal + shift
        while not all(abs(i) < np.pi for i in shifted):
            shifted[shifted > np.pi] -= 2 * np.pi
            shifted[shifted < -np.pi] += 2 * np.pi
        distances = data - shifted
        return distances

    def plotCalibration(self, fn=None):
        """
        Plot the result of the calibation and oscillation

        Parameters
        ----------
        fn: str, optional
            filename of the calibration trace plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        radii = [x[2] for x in self.calibration]
        mr = np.mean(radii)
        weights = 1
        mps = [[x[0], x[1]] for x in self.calibration]
        x, y, r, res = self.leastsq_circle(*zip(*mps), w=weights)
        circle_fit = plt.Circle((x, y), r, lw=1, facecolor='none', edgecolor='red')
        ax.add_artist(circle_fit)

        for i, mpt in enumerate(mps):
            circle = plt.Circle((mpt[0], mpt[1]), np.abs(mr - radii[i]) / 5, lw=1, alpha=0.3)
            ax.add_artist(circle)

        ax.plot(*zip(*mps), marker='x')
        mps_new = mps = [[x[0], x[1]] for x in self.newCalib]
        ax.plot(*zip(*mps_new), marker='x')

        ax.set_aspect(1)
        if fn is None:
            fig.savefig(__location__ + '/../img/out/calibrationTrace.png')
        else:
            fig.savefig(__location__ + '/../img/out/calibrationTrace_' + str(fn) + '.png')
