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
        self.fn = fn
        print('Reading image', fn)
        self.image = ndimage.imread(self.fn, flatten=True)
        print('Image loaded')
        self.dimensions = np.shape(self.image)
        self.dimy, self.dimx = self.dimensions

    def profileLine(self, p0, p1, img=None, interpolationOrder=1):
        # Generate an interpolated profile of the image matrix between points
        # p0, p1 with a resolution defined by the distance between the points
        # to avoid unnecessary interpolation where no information exists

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
        # Creates a transformed image where a sector is mapped to r/phi coordinates
        # The matrix is interpolated along the angled measurement lines
        #
        # r: Radius relative to the recorded image within the sector is recorded
        # dr: Distance from edge of image to geometric center of spiral
        # resolution: number of measurement lines. If None, the y-dimension of the image will be used
        # interpolationOrder: Polynomial order of interpolation algorithm between two pixels
        # plot: Plot the transformed image

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

    def binaryThresholding(self, matrix, thresh):
        # Threshold float matrix at the thresh value
        # Returns a binary matrix, usable for morphology
        binary = np.copy(matrix)
        binary[binary <= thresh] = False
        binary[binary > thresh] = True
        return binary

    # ******************************************************************************************
    # ******************************************************************************************

    # ******************************************************************************************
    # ******************************************************************************************

    def detectFeatures(self, matrix, plot=False):
        # Distinguish band from background in binary matrix
        print('Convolving')
        kernel = [1, -2, 1]
        #gaussian = ndimage.gaussian_filter1d(matrix, sigma=3, axis=1)
        # laplacian = np.abs(ndimage.convolve1d(gaussian, kernel, axis=1))
        equ0 = cv2.equalizeHist(matrix.astype('uint8'))
        laplacian0 = np.abs(ndimage.prewitt(matrix, axis=1))

        # morph_lap = np.abs(ndimage.morphological_laplace(matrix, size=3))

        equ = cv2.equalizeHist(laplacian0.astype('uint8'))
        equ = equ.astype('float32')
        lowpass = 0.5
        equ[equ < lowpass * np.max(equ)] = 0

        laplacian = laplacian0 * (1.0 / equ.max())
        laplacian0 = ndimage.morphology.black_tophat(laplacian0, size=11)

        # Threshold to cancel noise outside the band region
        print('Thresholding')
        thresh = 0.05
        #binary = self.binaryThresholding(laplacian, thresh)
        binary = ndimage.morphology.binary_opening(equ, iterations=2)
        binary = ndimage.morphology.binary_closing(binary, iterations=2)

        print('Binary Closing')
        # Closing to obtain continuous edge
        """
        structure = np.ones((11, 3))
        mask = ndimage.morphology.binary_closing(binary, iterations=1, structure=structure)
        # Crop mask to significant boundary
        mask_cropped = mask[~np.all(mask == 0, axis=1)]
        # Number of lost angles
        loss = np.shape(matrix)[0] - np.shape(mask_cropped)[0]
        """
        mask_cropped = binary
        # binary_fill_holes closes all holes that are completely surrounded by True values
        # Since it does not close to the boundary of the image, artificial True paddings are
        # added. Upped and lower padding have to be added separately to avoid filling of the
        # Gap between bands.

        # Add True boundary to bottom to close lower edgmaske holes
        print('Filling')
        mask_low = np.vstack((binary, np.ones((1, np.shape(binary)[1]))))
        mask_lowFill = ndimage.morphology.binary_fill_holes(mask_low)[:-1]

        # Repeat for top boundary
        mask_high = np.vstack((np.ones((1, np.shape(binary)[1])), mask_lowFill))
        mask_allFill = ndimage.morphology.binary_fill_holes(mask_high)[1:]

        mask_allFill = ndimage.morphology.binary_fill_holes(mask_cropped)
        mask_inv = 1 - mask_allFill
        mask_final = 1 - ndimage.morphology.binary_fill_holes(mask_inv)
        """
        print('Logical rolling')
        notLeft = np.roll(mask_final, 1, axis=0)
        # notRight = 1 - np.roll(mask_final, -1, axis=0)
        notTop = np.roll(mask_final, 1, axis=1)
        # notBottom = 1 - np.roll(mask_final, -1, axis=1)
        edges = (np.logical_xor(notLeft, mask_final) +
                 np.logical_xor(notTop, mask_final)
                 )

        structure = np.ones((3, 3))
        labeled_edges, num_features = ndimage.label(edges[1:-1], structure=structure)
        print(num_features)
        unique, counts = np.unique(labeled_edges, return_counts=True)
        bands = []
        for u, c in zip(unique, counts):
            if c > 100 and u != 0:
                band = np.argwhere(labeled_edges == u)
                bands.append(band)

        print(bands)
        """
        if plot:
            print('Plotting')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(laplacian)
            ax.set_xlabel('Radius [px]')
            ax.set_ylabel('Angle [idx]')
            fig.savefig('img/out/filter.png', dpi=600, interpolation='none')

        return mask_final  # , loss
