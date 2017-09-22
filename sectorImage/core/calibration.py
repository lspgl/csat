import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import math


class Calibrator:

    def __init__(self, fns):
        self.fns = fns
        print('Reading images', fns)
        self.images = [ndimage.imread(fn, flatten=True) for fn in self.fns]

    def houghTransform(self, im):
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gaussian = ndimage.gaussian_filter(im, sigma=5)
        derivative = np.abs(ndimage.prewitt(gaussian, axis=0))
        ret, thresh1 = cv2.threshold(derivative.astype('uint8'), 15, 10, cv2.THRESH_BINARY)
        op = ndimage.morphology.binary_opening(thresh1, iterations=2)
        op = op.astype('uint8')
        draw = (im * 0.5).astype('uint8')
        minLineLength = 1000
        maxLineGap = 200
        lines = cv2.HoughLinesP(op, 1, np.pi / 180, 1000, minLineLength, maxLineGap)
        print(len(lines))
        R_matrix = np.array([[0., 0.], [0., 0.]])
        q_vector = np.array([[0.], [0.]])
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(draw, (x1, y1), (x2, y2), 500, 2)
                r, q = self.p2vect(float(x1), float(y1), float(x2), float(y2))
                R_matrix += r
                q_vector += q

        print(R_matrix)
        print(q_vector)

        lsq = np.linalg.lstsq(R_matrix, q_vector)[0]
        print(lsq)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(draw)
        ax.scatter(*lsq, s=5, marker='x', color='red', lw=0.5)
        fig.savefig('img/out/houghTransform.jpg', dpi=600)

    def p2vect(self, x1, y1, x2, y2):
        # 2 points to m-q-line
        nx_raw = x2 - x1
        ny_raw = y2 - y1
        length = math.sqrt(nx_raw**2 + ny_raw**2)
        nx = nx_raw / length
        ny = ny_raw / length

        nx2 = nx**2
        ny2 = ny**2
        nxy = nx * ny
        r = np.array([[1 - nx2, -nxy], [-nxy, 1 - ny2]])
        q = np.array([[(1 - nx2) * x1 - nxy * y1], [-nxy * x1 + (1 - ny2) * y1]])
        return(r, q)


if __name__ == '__main__':
    fns = ['img/src/calibration/cpt1.jpg']
    c = Calibrator(fns)
    c.houghTransform(c.images[0])
