import numpy as np
import time
from scipy import ndimage
from .toolkit import vectools
from .toolkit.colors import Colors as _C
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2

import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Image:

    def __init__(self, fn, calibration, lock=None):
        """
        Image processing class

        Parameters
        ----------
        fn: string
            filename of the image to be processed
        """
        # t0 = time.time()
        self.fn = fn
        self.fn_npy = self.fn.split('.')[0] + '.npy'
        self.id = int(self.fn.split('cpt')[-1].split('.')[0])
        self.calibration = calibration
        # calibration_path = __location__ + '/../data/calibration.npy'
        # calibration = np.load(calibration_path)
        self.midpoint = calibration[self.id - 1][:-1]
        # self.midpoint = calibration[0][:-1]
        print(_C.YEL + 'Processing image ' + _C.BOLD + fn + _C.ENDC)
        if lock is not None:
            with lock:
                self.image = np.load(self.fn_npy)
                # self.image = cv2.imread(self.fn, cv2.IMREAD_GRAYSCALE)
        else:
            # self.image = cv2.imread(self.fn, cv2.IMREAD_GRAYSCALE)
            self.image = np.load(self.fn_npy)
        # self.image = np.rot90(self.image)
        # print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        self.dimensions = np.shape(self.image)
        self.dimy, self.dimx = self.dimensions

    def transformRadial(self, env=None, midpoint=None, plot=False):
        """
        Creates a transformed image where a sector is mapped to r/phi coordinates

        Parameters
        ----------
        midpoint: 2-tuple of floats, optional
            Origin of the polar coordinate system. If None is given, the calibration data from the current class instantation is taken
        plot: bool, optional
            Plot the transformed image. Default is False
            Cannot be used if multiprocessing is active

        Returns
        -------
        transformed: 2D array
            Coordinate transformed image
        angles: 1D array of floats
            angles between which the image is fully covered
        radii: 1D array of float
            distance scaling of rmax in the transformed image
        """
        r = self.dimx
        if midpoint is None:
            midpoint = self.midpoint
        # t0 = time.time()
        dr = midpoint[0] - self.dimx
        rmax = r + dr

        hplus = midpoint[1]
        hminus = self.dimy - midpoint[1]

        thetaPlus = -math.asin(hplus / rmax)
        thetaMinus = math.asin(hminus / rmax)

        # thetaPlus, thetaMinus = -thetaMinus, -thetaPlus

        thetaPlus_idx = int((thetaPlus + np.pi) / (2 * np.pi) * self.dimy)
        thetaMinus_idx = int((thetaMinus + np.pi) / (2 * np.pi) * self.dimy)

        # c = tuple(midpoint)
        cx, cy = midpoint
        c = (cx, cy)

        transformed = cv2.linearPolar(self.image, c, rmax, cv2.WARP_FILL_OUTLIERS)

        # Destroy the image object to free memory
        del self.image

        angles = np.linspace(thetaPlus, thetaMinus, thetaMinus_idx - thetaPlus_idx, endpoint=True)
        radii = np.linspace(0, rmax, self.dimx)

        self.dimensions = np.shape(transformed)
        self.dimy, self.dimx = self.dimensions
        absoluteZero = (self.dimy / 2 - thetaPlus_idx) - 1
        transformed = transformed[thetaPlus_idx:thetaMinus_idx]

        # Pad the transformed image with the boundary value
        start_idx = np.argmax(transformed > 0, axis=1)
        start_idx = np.ones((len(transformed)), np.uint8) * 1000
        transformed[:, :999] = 0
        for i in range(len(transformed)):
            transformed[i][transformed[i] == 0] = transformed[i, start_idx[i]]

        # Remove Calibration features
        calib_size_px = np.mean(np.array([x[2] for x in self.calibration]))
        calib_size_mm = env.calib_size_mm  # Outer radius of calibration piece
        tolerance = 1.1
        calib_width_mm = env.calib_width_mm * tolerance  # Width of the calibration piece
        # pitch_mm = self.env.pitch_mm  # Nominal electrode pitch
        scale = calib_size_mm / calib_size_px
        calibrationCutoff = (calib_size_mm - calib_width_mm) / scale * r / rmax
        # pitch = pitch_mm / scale
        transformed[:, int(calibrationCutoff):] = 0
        for i in range(len(transformed)):
            transformed[i][transformed[i] == 0] = transformed[i, int(calibrationCutoff) - 1]

        # plot = True

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(transformed)
            ax.axhline(y=absoluteZero)
            ax.set_aspect('auto')
            fig.savefig(__location__ + '/../img/out/cv2transform.png', dpi=300)
        # print('Coordinate transformation completed in ', str(round(time.time() - t0, 2)), 's')
        return transformed, angles, radii

    def detectFeatures(self, matrix, thresh_std=.5, plot=False):
        """
        Distinguish band from background in binary matrix

        Parameters
        ----------
        matrix: 2D array
            8Bit single channel source matrix to be processed
        plot: bool, optional
            Plot the transformed image. Default is False
            Cannot be used if multiprocessing is active

        Returns
        -------
        proc: 2D array
            Processed binary image

        """
        # t0 = time.time()
        start_range = 2000
        # end_range = np.shape(matrix)[1] - start_range
        # Initializing Empty array in Memory
        proc = np.empty(np.shape(matrix))
        start_search = np.empty(np.shape(matrix))[:, :start_range]
        end_search = np.empty(np.shape(matrix))[:, start_range:]
        matrix = matrix.astype(np.float64, copy=False)
        # print('Blurring')
        # Gaussian Blur to remove fast features

        cv2.GaussianBlur(src=matrix, ksize=(15, 3), dst=proc, sigmaX=1.5, sigmaY=10)
        ndimage.maximum_filter(proc, size=(5, 15), output=proc)
        # cv2.GaussianBlur(src=matrix[:, :start_range], ksize=(3, 0), dst=start_search, sigmaX=0, sigmaY=3)

        # cv2.GaussianBlur(src=matrix[:, start_range:], ksize=(31, 11), dst=end_search, sigmaX=0, sigmaY=0.1)
        start_search = matrix[:, :start_range]
        end_search = matrix[:, start_range:]
        # print('Convolving')
        # Convolving with Prewitt kernel in x-direction
        prewitt_kernel_x = np.tile([-1, 0, 1], (15, 1))
        # prewitt_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # print(prewitt_kernel_x)
        # prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y_width = 15
        prewitt_kernel_y = np.array([[1] * kernel_y_width, [0] *
                                     kernel_y_width, [-1] * kernel_y_width])

        # prewitt_kernel_y_element = np.tile(np.ones(kernel_y_width), (15, 1))
        # prewitt_kernel_y_end = np.concatenate((prewitt_kernel_y_element, np.zeros(15), -1 * prewitt_kernel_y_element))
        # print(prewitt_kernel_y_end)

        # print(prewitt_kernel_y_end)
        cv2.threshold(src=start_search, dst=start_search, thresh=20, maxval=255, type=cv2.THRESH_TOZERO)
        ndimage.minimum_filter(start_search, size=(15, 15), output=start_search)
        cv2.GaussianBlur(src=start_search, ksize=(11, 0), dst=start_search, sigmaX=0, sigmaY=5)
        cv2.filter2D(src=start_search, kernel=prewitt_kernel_y, dst=start_search, ddepth=-1)

        cv2.threshold(src=end_search, dst=end_search, thresh=20, maxval=255, type=cv2.THRESH_TOZERO)
        ndimage.minimum_filter(end_search, size=(15, 15), output=end_search)
        cv2.GaussianBlur(src=end_search, ksize=(11, 11), dst=end_search, sigmaX=0, sigmaY=5)
        cv2.filter2D(src=end_search, kernel=prewitt_kernel_y, dst=end_search, ddepth=-1)

        cv2.filter2D(src=proc, kernel=prewitt_kernel_x, dst=proc, ddepth=-1)
        cv2.GaussianBlur(src=proc, ksize=(11, 3), dst=proc, sigmaX=0, sigmaY=5)

        np.abs(start_search, out=start_search)
        np.abs(end_search, out=end_search)

        np.abs(proc, out=proc)

        start_amp = start_search.max()
        start_idx = np.unravel_index(start_search.argmax(), start_search.shape)
        start = (start_idx, start_amp)

        end_amp = end_search.max()
        end_idx = np.unravel_index(end_search.argmax(), end_search.shape)
        ex, ey = end_idx
        if ey != 0:
            ey += start_range
        end_idx = (ex, ey)
        end = (end_idx, end_amp)

        del start_search
        del end_search
        # print('Thresholding')
        proc_mean = np.mean(proc)
        proc_std = np.std(proc)

        thresh = proc_mean + thresh_std * proc_std
        thresh = 50.0
        # thresh = proc_mean
        # thresh = 0.1

        cv2.threshold(src=proc, dst=proc, thresh=thresh, maxval=1, type=cv2.THRESH_BINARY)

        proc = proc.astype(np.uint8, copy=False)

        # print('Connecting')
        # Label the complement regions of the binary image
        proc_inv = 1 - proc
        n_labels, labels, l_stats, l_centroids = cv2.connectedComponentsWithStats(image=proc_inv, connectivity=4)
        # The maximum number of pixels in a noise field
        # Everything larger is considered to be background
        fieldsize = 2e4
        # Label background fields
        gaps = []
        for i, stat in enumerate(l_stats):
            if stat[-1] > fieldsize:
                gaps.append(i)

        # Set background fields to zero
        for gap in gaps:
            labels[labels == gap] = 0
        # Set all forground fields to one
        labels[labels != 0] = 1
        labels = labels.astype(np.uint8, copy=False)

        # Combine foreground noise with with thresholded image
        cv2.bitwise_or(src1=proc, src2=labels, dst=proc)
        filtered = np.copy(proc)
        # plot = True
        if plot:
            print('Plotting')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(filtered)
            # ax.plot(filtered[0])
            ax.set_aspect('auto')
            ax.set_xlabel('Radius [px]')
            ax.set_ylabel('Angle [idx]')
            fig.savefig(__location__ + '/../img/out/filter' + str(self.id) + '.png', dpi=300, interpolation='none')

        # print('Features detected in', str(round(time.time() - t0, 2)), 's')
        return proc, (start, end)
