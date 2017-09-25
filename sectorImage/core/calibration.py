import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from operator import attrgetter

from toolkit.line import Line


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
        gaussian = ndimage.gaussian_filter(im, sigma=5)
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
            draw = (op * 50).astype('uint8')

        # Get probabilistic Hough transformation
        print('Computing Hough transform')
        minLineLength = 10
        maxLineGap = 500
        lines = cv2.HoughLinesP(op, 1, np.pi / 180, 100, minLineLength, maxLineGap)
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
        lines = []
        ms = []
        qs = []
        for line_raw in lines_raw:
            line = Line(*line_raw[0])
            lines.append(line)
            ms.append(line.m)
            qs.append(line.q)

        m_bin, m_edges = np.histogram(ms, bins='auto')
        avg_lines = []
        for i, b in enumerate(m_bin):
            if b >= sorted(m_bin, reverse=True)[3] and b > 0:
                ln_bin = []
                e_low = m_edges[i]
                e_high = m_edges[i + 1]
                for ln in lines:
                    if ln.m >= e_low and ln.m <= e_high:
                        ln_bin.append(ln)
                avg_lines.append(self.averageLines(ln_bin))

        return avg_lines

    def averageLines(self, line_bin):
        x1 = line_bin[0].x1
        x2 = line_bin[0].x2

        avg_m = sum(line.m for line in line_bin) / float(len(line_bin))
        correction_vect = [3000, avg_m * 3000]
        corrected_lines = []
        for ln in line_bin:
            dx, dy = correction_vect
            x1 = ln.x1
            y1 = ln.y1
            x2 = x1 + dx
            y2 = y1 + dy
            ln_new = Line(x1, y1, x2, y2)
            corrected_lines.append(ln_new)
        corrected_lines.sort(key=lambda x: x.q)
        delta_q = []
        for i, ln in enumerate(corrected_lines[:-1]):
            delta = corrected_lines[i + 1].q - ln.q
            delta_q.append(delta)

        interface = delta_q.index(max(delta_q)) + 1
        block_low = corrected_lines[:interface]
        block_high = corrected_lines[interface:]
        # avg_q = sum(line.q for line in line_bin) / float(len(line_bin))
        # avg_q = (line_bin[0].q + line_bin[-1].q) / 2
        # avg_q = (max(corrected_lines, key=attrgetter('q')).q + min(corrected_lines, key=attrgetter('q')).q) / 2
        avg_q_low = sum(line.q for line in block_low) / float(len(block_low))
        avg_q_high = sum(line.q for line in block_high) / float(len(block_high))
        avg_q = (avg_q_low + avg_q_high) / 2
        y1 = avg_m * x1 + avg_q
        y2 = avg_m * x2 + avg_q

        c = len(line_bin)
        print(c)

        return (Line(x1, y1, x2, y2), c)


if __name__ == '__main__':

    fns = ['img/src/calibration/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    fns = ['img/src/calibration/cpt3.jpg']
    # fns = ['../hardware/cpt' + str(i) + '.jpg' for i in range(1, 17)]

    c = Calibrator(fns)
    mps = c.getMidpoints(mp_FLAG=False)
    # mps = c.loadMidpoints()
    c.plotMidpoints(mps)
