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
        self.fns = fns
        self.mpflag = mpflag

    def computeMidpoint(self, fn, plot=False, smoothing=True, subsampling=False, lock=None):
        # t0 = time.time()
        fn_npy = fn.split('.')[0] + '.npy'
        print(_C.LIGHT + 'Calibrating image ' + _C.BOLD + fn + _C.ENDC)
        if lock is not None:
            with lock:
                src = np.load(fn_npy)
        else:
            src = np.load(fn_npy)

        src = np.rot90(src)
        # print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        src = src.astype(np.uint8, copy=False)
        # print('Blurring')
        im = np.empty(np.shape(src), np.uint8)
        # Gaussian Blur to remove fast features
        kernelsize = 5
        cv2.GaussianBlur(src=src, ksize=(kernelsize, kernelsize), dst=im, sigmaX=1.5, sigmaY=1.5)

        # print('Convolving')
        # Convolving with kernel
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        im = im.astype(np.int16, copy=False)
        cv2.threshold(src=im, dst=im, thresh=150, maxval=255, type=cv2.THRESH_TOZERO)[1]
        cv2.filter2D(src=im, kernel=prewitt_kernel_x, dst=im, ddepth=-1)
        np.abs(im, out=im)

        # print('Thresholding')
        mean_val = np.mean(im)
        std_val = np.std(im)
        thresh = mean_val + 3 * std_val
        cv2.threshold(src=im, dst=im, thresh=thresh, maxval=1, type=cv2.THRESH_BINARY)
        pt_x = [np.argmax(line > 0) for i, line in enumerate(im)]
        pt_y = np.arange(0, len(pt_x))
        pt_y = np.array([y for i, y in enumerate(pt_y) if pt_x[i] != 0])
        pt_x = np.array([x for x in pt_x if x != 0])

        if subsampling:
            subregion_size = 10
            n_regions = 100
            pt_x_sub, centroids = self.subsampling(pt_x, subregion_size, n_regions)

        # plot = True
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pt_x, pt_y, lw=.2, color='orange')
            if subsampling:
                ax.scatter(pt_x_sub, centroids, lw=.1, color='green', marker='o', s=1)
            ax.imshow(im)
            ax.set_xlim([700, 1200])
            ax.set_aspect('auto')
            fig.savefig(__location__ + '/../img/out/calibration_new_' +
                        fn.split('.')[0].split('/')[-1] + '.png', dpi=300)

        if subsampling:
            data = [pt_x_sub, centroids]
        else:
            data = [pt_x, pt_y]

        if smoothing:
            sigma = 11
            data = self.smoothing(data, sigma)

        return data

    def subsampling(self, x, n, size):
        centroids = np.linspace(size // 2, len(x - size // 2), num=n, endpoint=True)
        x_sub = np.empty(len(centroids))
        for i, c in enumerate(centroids):
            region = x[int(c) - size // 2:int(c) + size // 2]
            x_sub[i] = region.mean()

        return (x_sub, centroids)

    def smoothing(self, data, sigma):
        x, y = data
        filtered = ndimage.gaussian_filter1d(x, sigma)
        filtered[:2 * sigma] = x[:2 * sigma]
        filtered[-2 * sigma:] = x[-2 * sigma:]
        dnew = filtered, y

        plot = False
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(filtered, lw=0.5)
            #ax.plot(x, lw=0.5)
            # ax.set_xlim([900, 2500])
            fig.savefig('smooting_debug.png', dpi=300)
        return dnew

    def correction(self):
        print(_C.MAGENTA + 'Self-correcting calibration' + _C.ENDC)
        r = [c[2] for c in self.comp]
        r_mean = np.array(r).mean()
        r_std = np.array(r).std()
        print(r_std)
        out = []
        for c in self.comp:
            x = c[3]
            y = c[4]
            xc, yc, R, residu = self.leastsq_circle(x, y, w=1, fixedR=r_mean)
            out.append([xc, yc, R])
        return out

    def calc_R(self, x, y, xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_odr_multivariate(self, beta, x):
        x_flat = [x[i * 4000:(i + 1) * 4000] for i in range(16)]
        # delta = np.array([(d - beta[0][i])**2 + (self.grid - beta[1][i])**2 -
        # beta[2]**2 for i, d in enumerate(x_flat)])
        delta = np.array([np.abs(np.sqrt((d - beta[i * 2])**2 +
                                         (self.grid - beta[i * 2 + 1]) ** 2) -
                                 beta[-1]) for i, d in enumerate(x_flat)])
        delta = np.ndarray.flatten(delta)
        return delta

    def calc_estimate_odr_multivariate(self, dta):
        data = [dta.x[i * 4000:(i + 1) * 4000] for i in range(16)]
        # data = dta.x
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
        print('Computing orthogonal distance regression')
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
        center_estimates = np.empty(0)
        for i, d in enumerate(data):
            x, y = d
            x_m = np.mean(x)
            y_m = np.mean(y)
            mp = np.array([x_m, y_m])
            center_estimates = np.append(center_estimates, mp)
        opt = optimize.least_squares(self.f_multivariate, center_estimates, args=data, jac='3-point')

        calibration = []
        Rs = []
        for i in range(len(opt.x) // 2):
            xc = opt.x[i * 2]
            yc = opt.x[i * 2 + 1]
            center = [xc, yc]
            x, y = data[i]
            Ri = self.calc_R(x, y, *center)
            R = Ri.mean()
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
        lock = mp.Lock()
        self.comp = Parmap(self.computeMidpoint, self.fns, lock=lock)
        for c in self.comp:
            self.smoothing(c)
        # self.calibration = self.optimize_odr(self.comp)
        self.calibration = self.leastsq_circle_multivariate(self.comp)
        # print(self.calibration)
        # self.calibration = [c[:3] for c in self.comp]
        # self.calibration = self.correction()
        if tofile:
            np.save(__location__ + '/../data/calibration.npy', np.array(self.calibration))
        return np.array(self.calibration)

    def loadCalibration(self, fn):
        self.calibration = np.load(fn)
        return self.calibration

    def oscillationCircle(self):
        mps = [np.array([x[0], x[1]]) for x in self.calibration]
        # ravg = np.mean(np.array([x[2] for x in self.calibration]))
        radii = np.array([x[2] for x in self.calibration])
        x, y, r, res = self.leastsq_circle(*zip(*mps))
        mp = np.array([x, y])
        mps_zero = mps - mp
        mps_radial = np.array([np.array([np.sqrt(pt[0]**2 + pt[1]**2), np.arctan2(pt[1], pt[0])]) for pt in mps_zero])
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
        shifted = equal + shift
        while not all(abs(i) < np.pi for i in shifted):
            shifted[shifted > np.pi] -= 2 * np.pi
            shifted[shifted < -np.pi] += 2 * np.pi
        distances = data - shifted
        return distances

    def plotCalibration(self, fn=None):
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
