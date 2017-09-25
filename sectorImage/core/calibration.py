import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import math
import multiprocessing as mp


class Calibrator:

    def __init__(self, fns):
        self.fns = fns
        # self.images = [ndimage.imread(fn, flatten=True) for fn in self.fns]

    def getMidpoints(self, save=True):
        mp.set_start_method('spawn')
        ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        mps = pool.map(self.houghTransform, self.fns)
        #mps = [self.houghTransform(fn, plot=False) for fn in self.fns]
        print(mps)
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
        ret, thresh1 = cv2.threshold(derivative.astype('uint8'), 15, 10, cv2.THRESH_BINARY)
        # Morphological opening
        print('Morphology')
        op = ndimage.morphology.binary_opening(thresh1, iterations=2)
        # Convert to int
        op = op.astype('uint8')
        if plot:
            draw = (im * 0.5).astype('uint8')

        # Get probabilistic Hough transformation
        print('Computing Hough transform')
        minLineLength = 1000
        maxLineGap = 200
        lines = cv2.HoughLinesP(op, 1, np.pi / 180, 1000, minLineLength, maxLineGap)

        # Initialize R and q for least squares crossing point estimation
        R_matrix = np.array([[0., 0.], [0., 0.]])
        q_vector = np.array([[0.], [0.]])

        # Iterate over detected Hough lines
        print('Computing matrix elements')
        for line in lines:
            for x1, y1, x2, y2 in line:
                if plot:
                    cv2.line(draw, (x1, y1), (x2, y2), 500, 2)
                # Convert start and end point of line to matrix and vector elements
                r, q = self.p2vect(float(x1), float(y1), float(x2), float(y2))
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

        return lsq_tuple

    def p2vect(self, x1, y1, x2, y2):
        # Calculate matrix and vector elements for point set
        # a = (x1,y1)T
        # n = ||1->2||T
        # r = I - nnT
        # q = r*a

        nx_raw = x2 - x1
        ny_raw = y2 - y1
        # Normalize
        length = math.sqrt(nx_raw**2 + ny_raw**2)
        nx = nx_raw / length
        ny = ny_raw / length

        nx2 = nx**2
        ny2 = ny**2
        nxy = nx * ny
        # Generate matrix
        r = np.array([[1 - nx2, -nxy], [-nxy, 1 - ny2]])
        # Generate vector
        q = np.array([[(1 - nx2) * x1 - nxy * y1], [-nxy * x1 + (1 - ny2) * y1]])
        return(r, q)


if __name__ == '__main__':
    fns = ['img/src/calibration/cpt1.jpg']
    fns = ['img/src/calibration/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    fns = ['../hardware/cpt' + str(i) + '.jpg' for i in range(1, 17)]

    c = Calibrator(fns)
    mps = c.getMidpoints()
    # mps = c.loadMidpoints()
    c.plotMidpoints(mps)
