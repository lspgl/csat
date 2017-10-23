import numpy as np
import time
from scipy import ndimage
from .toolkit import vectools
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2
# from skimage.filters import threshold_local


class Image:

    def __init__(self, fn):
        """
        Image processing class

        Parameters
        ----------
        fn: string
            filename of the image to be processed
        """
        t0 = time.time()
        self.fn = fn
        print('Reading image', fn)
        self.image = cv2.imread(self.fn, cv2.IMREAD_GRAYSCALE)
        self.image = np.rot90(self.image)
        print('Image loaded in', str(round(time.time() - t0, 2)), 's')
        self.dimensions = np.shape(self.image)
        self.dimy, self.dimx = self.dimensions

    def profileLine(self, p0, p1, img=None, interpolationOrder=1):
        """
        Generate an interpolated profile of the image matrix between 2 points

        Parameters
        ----------
        p0, p1: tuples of floats
            two points (x,y) between which to interpolate
        img: 2D array, optional
            source image. If none, the image specified during class instantation is used. Default is None
        interpolationOrder: int, optional
            order of the interpolation polynom for the coordinate transform. Default is 1
        """

        if img is None:
            img = self.image

        x0, y0 = p0
        x1, y1 = p1
        res = vectools.pointdist(p0, p1)
        res = int(res)
        x, y = np.linspace(x0, x1, res), np.linspace(y0, y1, res)

        zi = ndimage.map_coordinates(img, np.vstack((y, x)), order=interpolationOrder, mode='nearest')
        return zi

    def lineSweep(self, r, dr=0, resolution=None, interpolationOrder=1, plot=False):
        """
        Creates a transformed image where a sector is mapped to r/phi coordinates
        The matrix is interpolated along the angled measurement lines

        Parameters
        ----------
        r: float
            Radius relative to the recorded image within the sector is recorded
        dr: float, optional
            Distance from edge of image to geometric center of spiral. Default is 0
        resolution: int
            number of measurement lines. If None, the y-dimension of the image will be used
        interpolationOrder: int, optional
            Polynomial order of interpolation algorithm between two pixels. Default is 1
        plot: bool, optional
            Plot the transformed image. Default is False
            Cannot be used if multiprocessing is active

        Returns
        -------
        linesnew: 2D array
            Coordinate transformed image
        angles: 1D array of floats
            angles between which the image is sweeped
        """

        print('Sweeping line with radius', r)
        # Pre-Cropping image to maximum size defined by measurement radius
        croppedImage = self.image[:, self.dimx - r:]

        # Shift to true center point
        dr = 454
        rmax = r + dr

        # TODO: Test if this is the right way around
        cy = 2014
        hplus = cy
        hminus = 4000 - cy
        # hplus = 5.0
        # hminus = 6.0
        htot = hplus + hminus

        lplus = hplus / htot * self.dimy
        lminus = hminus / htot * self.dimy

        # Global center point for circular measurement
        # c0 = (r, self.dimy / 2)
        c0 = (r, lplus)
        # Maximum sector angle which can be covered by a given radius
        # theta = math.asin(self.dimy / (2.0 * r))  # Radians

        # thetaTrue = math.asin(self.dimy / (2 * rmax))
        thetaPlus = -math.asin(lplus / rmax)
        thetaMinus = math.asin(lminus / rmax)

        self.thetaCovered = np.abs(thetaPlus - thetaMinus)
        print('Angular Coverage: ' + str(round(self.thetaCovered * 180 / np.pi, 2)) + '°')

        # Number of measurement lines
        if resolution is None:
            resolution = self.dimy

        # Space of angles
        # angles = np.linspace(-theta, theta, resolution, endpoint=True)
        # angles = np.linspace(-thetaTrue, thetaTrue, resolution, endpoint=True)
        angles = np.linspace(thetaPlus, thetaMinus, resolution, endpoint=True)

        lines = []
        time0 = time.time()
        timecounter = 1
        reportInterval = 5
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # Iterate over angles
        for i, a in enumerate(angles):
            # print 'interpolating at theta', a, i
            # x,y coordinates in Pre-cropped coordinate system

            x = c0[0] - (rmax * math.cos(a) - dr)
            y = rmax * math.sin(a) + c0[1]

            dy = dr * math.tan(a)

            cshift = c0[1] + dy
            if plot:
                ax.plot([x, c0[0]], [y, cshift], lw=0.01, color='red')

            # Further crop image to ROI given by angle
            # Interpolation points are the corners of the resulting image
            if a <= 0:
                iterationImage = croppedImage[int(y):int(cshift), int(x):]
                cT = (np.shape(iterationImage)[1], np.shape(iterationImage)[0])
                pT = (0, 0)
            elif a > 0:
                iterationImage = croppedImage[int(cshift):int(y), int(x):]
                cT = (np.shape(iterationImage)[1], 0)
                pT = (0, np.shape(iterationImage)[0])

            # Don't interpolate in 1D lines
            if np.shape(iterationImage)[0] < 2:
                iterationImage = croppedImage[int(cshift):int(cshift) + 1, :]
                line_inverted = croppedImage[int(cshift), :]
                line = line_inverted[::-1]
            else:
                line = self.profileLine(cT, pT,
                                        img=iterationImage,
                                        interpolationOrder=interpolationOrder)

            lines.append(line)
            elapsed = time.time() - time0
            if int(elapsed) > 0 and elapsed > timecounter:
                if timecounter % reportInterval == 0:
                    print('Sweeped', i, '/', len(angles), '(', 100 * float(i) /
                          float(len(angles)), '% ) angles in', timecounter, 'seconds')
                timecounter += 1

        # Pad lines with zeros on inside
        linesnew = []
        for line in lines[:]:
            delta = rmax - len(line) + 1
            deltaArr = np.ones(int(delta)) * line[0]
            line = np.append(deltaArr, line)
            linesnew.append(line)

        if plot:
            fig.savefig('img/out/debugImage.png', dpi=300)
        print('Sweep complete for', len(angles), 'angles in', round((time.time() - time0), 2), 's')

        # print lines
        # plot = True
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(linesnew, aspect='auto')
            ax.set_xlabel('Radius [px]')
            ax.set_ylabel('Angle [idx]')
            fig.savefig('img/out/transformed.png', dpi=300)

        return np.array(linesnew), angles

    # ******************************************************************************************
    # ******************************************************************************************

    # ******************************************************************************************
    # ******************************************************************************************

    def detectFeatures(self, matrix, plot=False):
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
        t0 = time.time()

        # Initializing Empty array in Memory
        proc = np.empty(np.shape(matrix))

        print('Blurring')
        # Gaussian Blur to remove fast features
        cv2.GaussianBlur(src=matrix, ksize=(0, 3), dst=proc, sigmaX=3, sigmaY=0)

        print('Convolving')
        # Convolving with Prewitt kernel in x-direction
        prewitt_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        cv2.filter2D(src=proc, kernel=prewitt_kernel, dst=proc, ddepth=-1)
        np.abs(proc, out=proc)

        print('Thresholding')
        # Threshold to 30% for noise reduction
        proc *= proc * (1.0 / proc.max())
        thresh = 0.3
        cv2.threshold(src=proc, dst=proc, thresh=thresh, maxval=1, type=cv2.THRESH_BINARY)

        print('Morphology')
        # Increase Noise reduction through binary morphology
        morph_kernel = np.ones((5, 5), np.uint8)
        cv2.morphologyEx(src=proc, dst=proc, op=cv2.MORPH_OPEN, iterations=1, kernel=morph_kernel)
        cv2.morphologyEx(src=proc, dst=proc, op=cv2.MORPH_CLOSE, iterations=1, kernel=morph_kernel)

        proc = proc.astype(np.uint8, copy=False)

        print('Connecting')
        # Label the complement regions of the binary image
        proc_inv = 1 - proc
        n_labels, labels, l_stats, l_centroids = cv2.connectedComponentsWithStats(image=proc_inv, connectivity=4)
        # The maximum number of pixels in a noise field
        # Everything larger is considered to be background
        fieldsize = 1e5
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

        if plot:
            print('Plotting')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(proc)

            ax.set_xlabel('Radius [px]')
            ax.set_ylabel('Angle [idx]')
            fig.savefig('img/out/filter.png', dpi=600, interpolation='none')

        print('Features detected in', str(round(time.time() - t0, 2)), 's')
        return proc  # , loss
