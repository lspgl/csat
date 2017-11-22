import numpy as np
from numpy.linalg import eig, inv
import cv2
import time
import math
import matplotlib.pyplot as plt
from scipy import optimize
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

    def computeMidpoint(self, fn, plot=False, lock=None):
        # t0 = time.time()
        fn_npy = fn.split('.')[0] + '.npy'
        print(_C.LIGHT + 'Calibrating image ' + _C.BOLD + fn + _C.ENDC)
        if lock is not None:
            with lock:
                src = np.load(fn_npy)
                # src = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        else:
            # src = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            src = np.load(fn_npy)

        src = np.rot90(src)
        # print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        src = src.astype(np.uint8, copy=False)
        # print('Blurring')
        im = np.empty(np.shape(src), np.uint8)
        # Gaussian Blur to remove fast features
        cv2.GaussianBlur(src=src, ksize=(0, 5), dst=im, sigmaX=5, sigmaY=5)
        # cv2.equalizeHist(src=im, dst=im)

        # print('Convolving')
        # Convolving with kernel
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        im = im.astype(np.int16, copy=False)
        cv2.filter2D(src=im, kernel=prewitt_kernel_x, dst=im, ddepth=-1)
        # cv2.filter2D(src=im, kernel=prewitt_kernel_y, dst=im, ddepth=-1)
        np.abs(im, out=im)

        # print('Thresholding')
        thresh = .5
        cv2.threshold(src=im, dst=im, thresh=thresh * np.max(im), maxval=1, type=cv2.THRESH_BINARY)

        pt_x = [np.argmax(line > 0) for i, line in enumerate(im)]
        pt_y = np.arange(0, len(pt_x))
        border_region = 1000
        pt_y = [y for i, y in enumerate(pt_y) if pt_x[i] != 0 and pt_x[i] < border_region]
        pt_x = [x for x in pt_x if x != 0 and x < border_region]
        subregion_size = 10
        head = [pt_x[:subregion_size], pt_y[:subregion_size]]
        tail = [pt_x[-subregion_size:], pt_y[-subregion_size:]]
        ctr = [pt_x[len(pt_x) // 2 - (subregion_size // 2):len(pt_x) // 2 + (subregion_size // 2)],
               pt_y[len(pt_y) // 2 - (subregion_size // 2):len(pt_y) // 2 + (subregion_size // 2)]]
        head_avg = [np.mean(head[0]), np.mean(head[1])]
        tail_avg = [np.mean(tail[0]), np.mean(tail[1])]
        ctr_avg = [np.mean(ctr[0]), np.mean(ctr[1])]

        pt3_x = [head_avg[0], tail_avg[0], ctr_avg[0]]
        pt3_y = [head_avg[1], tail_avg[1], ctr_avg[1]]
        # xc_, yc_, r, residu = self.leastsq_circle(pt_x, pt_y)
        # xc, yc, r, residu = self.leastsq_circle(pt3_x, pt3_y)

        # print('...')
        # print([xc_, yc_])
        # print([xc, yc])
        """
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            circle = plt.Circle((xc, yc), r, facecolor='none', lw=.5, edgecolor='red')
            ax.scatter(xc, yc, marker='x', s=10)
            ax.scatter(pt_x, pt_y, lw=.1, color='orange', marker='o', s=.2)
            ax.scatter(pt3_x, pt3_y, lw=.1, color='green', marker='o', s=1)
            ax.imshow(im)
            # ax.set_xlim(280, border_region)
            ax.set_aspect('auto')
            ax.add_artist(circle)
            fig.savefig(__location__ + '/../img/out/calibration_new_' +
                        fn.split('.')[0].split('/')[-1] + '.png', dpi=300)
        """
        # return [xc, yc, r, pt3_x, pt3_y]
        return [pt3_x, pt3_y]

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
        delta = Rall - Rall.mean()
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

        print(calibration)
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
        self.calibration = self.leastsq_circle_multivariate(self.comp)
        print(self.calibration)
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
        ravg = np.mean(np.array([x[2] for x in self.calibration]))
        x, y, r, res = self.leastsq_circle(*zip(*mps))
        mp = np.array([x, y])
        # thetas = [math.atan2(*(np.flipud(mpi - mp))) for mpi in mps]
        thetas = np.zeros(len(mps))
        for i, mpi in enumerate(mps):
            theta = math.atan2(*(np.flipud(mpi - mp)))
            if theta < 0:
                theta += 2 * np.pi
            thetas[i] = theta

        thetas_equal = np.linspace(0, 2 * np.pi, num=len(mps), endpoint=True)
        dt = thetas - thetas_equal
        dt_avg = np.zeros(len(mps))
        for i in range(len(mps)):
            shifted = np.roll(thetas_equal, i)
            dt = np.abs(thetas - shifted)
            for j, d in enumerate(dt[:]):
                if d > np.pi:
                    dt[j] -= 2 * np.pi
                    dt[j] = abs(dt[j])
            dt_avg[i] = np.mean(np.square(dt))
        best_thetas_shifted = np.roll(thetas_equal, np.argmin(dt_avg))
        opt = optimize.least_squares(self.arrayDistance, 0, args=(thetas, best_thetas_shifted))['x']
        optshift = best_thetas_shifted + opt[0]

        newCalib = []

        for i in range(len(mps)):
            xnew = x + np.cos(thetas[i]) * r
            ynew = y + np.sin(thetas[i]) * r
            newCalib.append([xnew, ynew, ravg])

        circle_fit = plt.Circle((x, y), r, lw=1, facecolor='none', edgecolor='red')

        x0 = mps[0][0]
        y0 = mps[0][1]

        dx = x0 - x
        dy = y0 - y

        theta = math.atan2(dy, dx)

        oscillation = [x, y, r, theta]
        # print(oscillation)
        return oscillation, newCalib

    def arrayDistance(self, shift, data, equal):
        distances = data - (equal + shift)
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

        ax.set_aspect(1)
        if fn is None:
            fig.savefig(__location__ + '/../img/out/calibrationTrace.png')
        else:
            fig.savefig(__location__ + '/../img/out/calibrationTrace_' + str(fn) + '.png')
