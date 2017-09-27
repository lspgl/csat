import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import cv2
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from operator import attrgetter

from toolkit.line import Line
from toolkit.intersection import Intersection

import itertools


class Calibrator:

    def __init__(self, fns):
        self.fns = fns
        # self.images = [ndimage.imread(fn, flatten=True) for fn in self.fns]

    def getMidpoints(self, save=True, mp_FLAG=True):
        if mp_FLAG:
            mp.set_start_method('spawn')
            ncpus = mp.cpu_count()
            pool = mp.Pool(ncpus)
            payload = pool.map(self.houghTransform, self.fns)
        else:
            payload = [self.houghTransform(fn, plot=True) for fn in self.fns]
        mps = [p[0] for p in payload]
        self.i_pts = [p[1] for p in payload]
        self.f_pts = [p[2] for p in payload]
        # mps = [self.houghTransform(fn, plot=False) for fn in self.fns]
        if save:
            np.save('data/calibration.npy', mps)
        return mps

    def loadMidpoints(self, fn='data/calibration.npy'):
        mps = np.load(fn)
        print(mps)
        return mps

    def plotMidpoints(self, mps):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(*zip(*mps), ms=5, lw=0.5, marker='x')
        for i_pt, f_pt in zip(self.i_pts, self.f_pts):
            for i, f in zip(i_pt, f_pt):
                # ax.plot([i[0], f[0]], [i[1], f[1]], lw=0.1)
                pass
        fig.savefig('img/out/midpoints.png', dpi=300)

    def houghTransform(self, fn, plot=False):
        print('Reading image', fn)
        im = ndimage.imread(fn, flatten=True)
        # Blur out fast features
        print('Blurring')
        gaussian = ndimage.gaussian_filter(im, sigma=3)
        # Edge detection filter
        print('Prewitt Filtering')
        derivative = np.abs(ndimage.prewitt(gaussian, axis=0))

        # Binary thresholding
        print('Thresholding')
        ret, thresh1 = cv2.threshold(derivative.astype('uint8'), 5, 255, cv2.THRESH_BINARY)
        # Morphological opening
        print('Morphology')
        op = ndimage.morphology.binary_opening(thresh1, iterations=2)

        # Convert to int
        op = op.astype('uint8')
        if plot:
            draw = (im / np.max(im) * 100).astype('uint8')

        # Get probabilistic Hough transformation
        print('Computing Hough transform')
        minLineLength = 1500
        maxLineGap = 10
        lines = cv2.HoughLinesP(op, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
        print('Number of detected lines:', str(len(lines)))
        if plot:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(draw, (x1, y1), (x2, y2), 200, 1)
                    pass

        # Initialize R and q for least squares crossing point estimation
        R_matrix = np.array([[0., 0.], [0., 0.]])
        q_vector = np.array([[0.], [0.]])

        # Iterate over detected Hough lines
        i_pts = []
        f_pts = []
        print('Computing matrix elements')
        avg_lines = self.linePooling(lines)
        for ln in avg_lines:
            r, q, a, n = ln[0].p2vect(confidence=ln[1])
            if plot:
                cv2.line(draw, (int(ln[0].x1), int(ln[0].y1)), (int(ln[0].x2), int(ln[0].y2)), 255, 5)
            i_pts.append(a)
            f_pts.append([sum(x) for x in zip(a, n)])
            R_matrix += r
            q_vector += q

        # Run least square fitting for Rm = q to find midpoint m
        print('Computing least squares solution')
        lsq = np.linalg.lstsq(R_matrix, q_vector)[0]
        lsq_tuple = [lsq[0, 0], lsq[1, 0]]
        print('Inferred Midpoint at', lsq_tuple)

        if plot:
            print('Plotting')
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(draw)
            ax.scatter(*lsq, s=5, marker='x', color='red', lw=0.5)
            fig.savefig('img/out/houghTransform.jpg', dpi=600)

        return (lsq_tuple, i_pts, f_pts)

    def linePooling(self, lines_raw):
        # Compound detected edges into a single line along a ridge
        lines = []
        ms = []
        qs = []
        # Convert lines to Line objects
        for i, line_raw in enumerate(lines_raw):
            line = Line(*line_raw[0], identifier=i)
            lines.append(line)
            ms.append(line.m)
            qs.append(line.q)

        # Histogram along slope to distinguish ridges
        m_bin, m_edges = np.histogram(ms, bins='auto')
        avg_lines = []
        for i, b in enumerate(m_bin):
            # Take only the 4 largest ridges
            if b >= sorted(m_bin, reverse=True)[2] and b > 0:
                # Collect the lines in the histogram bins
                ln_bin = []
                e_low = m_edges[i]
                e_high = m_edges[i + 1]
                for ln in lines:
                    if ln.m >= e_low and ln.m <= e_high:
                        ln_bin.append(ln)
                # Compute the true position of the ridge
                # self.rejectLines(ln_bin)
                avg_lines.append(self.averageLines(ln_bin))

        return avg_lines

    def rejectLines(self, line_bin):
        print('Rejecting minor lines')
        pairs = itertools.combinations(line_bin, 2)
        for ln1, ln2 in pairs:
            intersect = Intersection(ln1, ln2)

    def averageLines(self, line_bin):
        # Take arbitrary start and end coordinates
        x1_0 = 0
        x2_0 = 7000

        # Global average of the slope in a ridge
        avg_correction_m = sum(line.m for line in line_bin) / float(len(line_bin))

        # Vector along the global slope
        correction_vect = [1, avg_correction_m]
        corrected_lines = []
        for i, ln in enumerate(line_bin):
            # Generate a list of edges, all pointing along the same slope
            dx, dy = correction_vect
            x1 = ln.x1
            y1 = ln.y1
            x2 = x1 + dx
            y2 = y1 + dy
            ln_new = Line(x1, y1, x2, y2, identifier=i)
            corrected_lines.append(ln_new)
        # Sort the list along the intercept
        corrected_lines.sort(key=lambda x: x.q)
        # Sort the original list with unmodified slope along the intercepts of the corrected slope
        id_order = [ln.identifier for ln in corrected_lines]
        line_bin_s = [line_bin[i] for i in id_order]

        # Get the change in intercept between two q-adjacent lines
        delta_q = []
        for i, ln in enumerate(corrected_lines[:-1]):
            delta = corrected_lines[i + 1].q - ln.q
            delta_q.append(delta)
        """
        qs = [ln.q for ln in corrected_lines]
        kernel = gaussian_kde(qs, bw_method=2e-1)
        x = np.linspace(qs[0], qs[-1], num=100)
        kde = kernel(x)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, kde)
        ax.scatter(qs, np.zeros(len(qs)))
        fig.savefig('img/out/intercepts_' + str(rand.randint(0, 4)) + '.png')
        """

        # The largest jump in q should differentiate the upper from the lower edge
        interface = delta_q.index(max(delta_q)) + 1
        block_low = line_bin_s[:interface]
        block_high = line_bin_s[interface:]

        # Get the xy values of all detected edges along an edge
        xs_low = [ln.x1 for ln in block_low] + [ln.x2 for ln in block_low]
        ys_low = [ln.y1 for ln in block_low] + [ln.y2 for ln in block_low]

        xs_high = [ln.x1 for ln in block_high] + [ln.x2 for ln in block_high]
        ys_high = [ln.y1 for ln in block_high] + [ln.y2 for ln in block_high]

        # Linearly interpolate along an edge
        m_low, q_low = np.polyfit(xs_low, ys_low, 1)
        m_high, q_high = np.polyfit(xs_high, ys_high, 1)

        # Equally weigh upper and lower edges (even if they don't have an equal amount of Hough lines)
        avg_m = (m_low + m_high) / 2
        avg_q = (q_low + q_high) / 2

        # Generate a line with along the ridge
        y1_0 = avg_m * x1_0 + avg_q
        y2_0 = avg_m * x2_0 + avg_q

        # The confidence for the least square fitting is the number of detected Hough lines
        c = len(line_bin)
        print(c)
        c = 1

        return (Line(x1_0, y1_0, x2_0, y2_0), c)


if __name__ == '__main__':

    fns = ['img/src/calibration/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    fns = ['img/src/calibration/cpt1.jpg']
    # fns = ['../hardware/cpt' + str(i) + '.jpg' for i in range(1, 17)]

    c = Calibrator(fns)
    mps = c.getMidpoints(mp_FLAG=False)
    # mps = c.loadMidpoints()
    c.plotMidpoints(mps)
