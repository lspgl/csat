import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
from scipy import optimize
import multiprocessing as mp
from .toolkit.parmap import Parmap

import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Calibrator:

    def __init__(self, fns, mpflag=True):
        self.fns = fns
        self.mpflag = mpflag

    def computeMidpoint(self, fn, plot=False):
        t0 = time.time()
        print('Reading image', fn)
        src = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        src = np.rot90(src)
        print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        src = src.astype(np.uint8, copy=False)
        print('Blurring')
        im = np.empty(np.shape(src), np.uint8)
        # Gaussian Blur to remove fast features
        cv2.GaussianBlur(src=src, ksize=(0, 5), dst=im, sigmaX=1, sigmaY=1)
        # cv2.equalizeHist(src=im, dst=im)

        print('Convolving')
        # Convolving with kernel
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        im = im.astype(np.int16, copy=False)
        cv2.filter2D(src=im, kernel=prewitt_kernel_x, dst=im, ddepth=-1)
        # cv2.filter2D(src=im, kernel=prewitt_kernel_y, dst=im, ddepth=-1)
        np.abs(im, out=im)

        print('Thresholding')
        thresh = .5
        cv2.threshold(src=im, dst=im, thresh=thresh * np.max(im), maxval=1, type=cv2.THRESH_BINARY)

        pt_x = [np.argmax(line > 0) for i, line in enumerate(im)]
        pt_y = np.arange(0, len(pt_x))
        border_region = 1000
        pt_y = [y for i, y in enumerate(pt_y) if pt_x[i] != 0 and pt_x[i] < border_region]
        pt_x = [x for x in pt_x if x != 0 and x < border_region]

        xc, yc, r, residu = self.leastsq_circle(pt_x, pt_y)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.imshow(im[im.shape[0] // 2 - im.shape[0] // 4:im.shape[0] // 2 + im.shape[0] // 4,
            #             im.shape[1] // 2 - im.shape[1] // 4:im.shape[1] // 2 + im.shape[1] // 4, ])
            circle = plt.Circle((xc, yc), r, facecolor='none', lw=.5, edgecolor='red')
            ax.scatter(xc, yc, marker='x', s=10)
            ax.scatter(pt_x, pt_y, lw=.1, color='orange', marker='o', s=1)

            ax.imshow(im)
            #ax.set_xlim(xc - 1.1 * r, xc + 1.1 * r)
            #ax.set_ylim(yc - 1.1 * r, yc + 1.1 * r)
            #ax.set_xlim(np.min(pt_x), np.max(pt_x))
            ax.set_xlim(280, 670)
            ax.set_aspect('auto')
            ax.add_artist(circle)
            fig.savefig(__location__ + '/../img/out/calibration_new.png', dpi=600)

        return [xc, yc, r, pt_x, pt_y]

    def correction(self):
        r = [c[2] for c in self.comp]
        r_mean = np.array(r).mean()
        out = []
        for c in self.comp:
            x = c[3]
            y = c[4]
            xc, yc, R, residu = self.leastsq_circle(x, y, w=1, fixedR=r_mean)
            out.append([xc, yc, R])

        return out

    def calc_R(self, x, y, xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f(self, c, x, y, w=1, fixedR=None):
        Ri = self.calc_R(x, y, *c)
        if fixedR is None:
            delta = Ri - Ri.mean()
        else:
            delta = Ri - fixedR
        delta_w = delta * w
        return delta_w

    def leastsq_circle(self, x, y, w=1, fixedR=None):
        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(self.f, center_estimate, args=(x, y, w, fixedR))
        xc, yc = center
        Ri = self.calc_R(x, y, *center)
        R = Ri.mean()
        residu = np.sum((Ri - R)**2)
        print('Residual:', residu)
        return (xc, yc, R, residu)

    def computeAll(self, tofile=True):
        self.comp = Parmap(self.computeMidpoint, self.fns)

        #self.comp = [self.computeMidpoint(fn) for fn in self.fns]
        self.calibration_raw = [c[:3] for c in self.comp]
        self.calibration = self.correction()
        if tofile:
            np.save(__location__ + '/../data/calibration.npy', np.array(self.calibration))
        return self.calibration

    def loadCalibration(self, fn):
        self.calibration = np.load(fn)
        print(self.calibration)
        return self.calibration

    def oscillationCircle(self):
        mps = [[x[0], x[1]] for x in self.calibration]
        x, y, r, res = self.leastsq_circle(*zip(*mps))
        x0 = mps[0][0]
        y0 = mps[0][1]

        dx = x0 - x
        dy = y0 - y

        theta = math.atan2(dy, dx)

        oscillation = [x, y, r, theta]
        print(oscillation)
        return oscillation

    def plotCalibration(self):
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

        ax.set_aspect(1)
        fig.savefig(__location__ + '/../img/out/calibrationTrace.png')
