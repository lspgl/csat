from . import singleImage
import numpy as np
import operator
from .toolkit import pickler
from .toolkit.parmap import Parmap
from .toolkit.colors import Colors as _C
from .toolkit import vectools
from .toolkit.segment import Segment
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Stitcher:

    def __init__(self, fns, calibration=None, env=None, mpflag=True):
        """
        Stitching class to combine multiple processed images

        Parameters
        ----------
        fns: List of strings
            filenames to be read and combined. The combination is done in the order of the supplied list
        calibration: List of floats
            returned from the calibration sequence
        env: Environment object
            container for data about the physical environment
        mpflag: bool, optional
            multiprocessing flag. If on, the image processing is distributed over the available cores.
            This disables plotting of individual images. Default is True
        """
        self.fns = fns
        self.images = []
        self.mpflag = mpflag

        if env is None:
            raise Exception('Missing Environment')
        self.env = env

        if calibration is None:
            print('Loading Calibration from File')
            calibration_path = __location__ + '/../data/calibration.npy'
            self.calibration = np.load(calibration_path)
        else:
            self.calibration = calibration

    def loadImages(self):
        """
        Initialize the SingleImage instances and process the images.
        """

        if self.mpflag:
            # mp.set_start_method('spawn')
            #ncpus = mp.cpu_count()
            #pool = mp.Pool(ncpus)
            self.images = Parmap(self.singleRoutine, self.fns, self.calibration)
            # pool.close()
            # pool.join()
        else:
            for fn in self.fns:
                # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
                im = singleImage.SingleImage(fn, self.calibration)
                # im.getFeatures(npz=npzfn)
                im.getFeatures()
                # im.setFeatures(npz=npzfn)
                im.getLines()
                self.images.append(im)

    def stitchImages(self, plot=True, tofile=True):
        print(_C.MAGENTA + 'Stitching images' + _C.ENDC)
        """
        Stitch the images together by using the calibrated midpoints and absolute angles

        Parameters
        ----------
        plot: bool, optional
            plot the output. Default True
        tofile: bool, optional
            save the band segment coordinates to file including the image and band index as meta data in a tuple

        Returns
        -------
        segments: list of 4-tuples
            Each tuple contains 2 lists with angles and radii, followed by image and band index.
            The band index is only relative to the current image, not to the overall spiral
        """
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        segments = []
        for i, image in enumerate(self.images[::-1]):
            stepsize_angles = (image.angles[-1] - image.angles[0]) / len(image.angles)
            stepsize_radii = (image.radii[-1] - image.radii[0]) / len(image.radii)
            # img_segs = []
            for j, coord in enumerate(zip(image.r, image.phi)):
                rs, phis = coord
                phis = (np.array(phis) * stepsize_angles) + image.angles[0] + (i * 2 * np.pi / len(self.fns))
                rs = (np.array(rs) * stepsize_radii)
                s = (phis, rs, i, j)
                segments.append(s)
                if plot:
                    ax.plot(phis, rs, lw=0.5)
            # segments.append(img_segs)
        if plot:
            ax.set_xlabel('Angle [rad]')
            ax.set_ylabel('Radius [px]')
            fig.savefig(__location__ + '/../img/out/stitched.png', dpi=300)

        if tofile:
            np.save(__location__ + '/../data/stitched.npy', segments)

        return segments

    def combineSegments(self, segments, plot=False):
        """
        Combine the segments of the stitched image to a continuous spiral

        Parameters
        ----------
        segments: list of 4-tuples
            see return of stitchImages()
        plot: bool, optional
            plot the output. Default False

        Returns
        -------
        parametrized: 3-tuple
            2 lists of angles and radii of the parametrized spiral followed by the calibrated relationship between pixels and mm
        """
        segments = [Segment(*coords, identity=i) for i, coords in enumerate(segments)]
        # Calibration piece
        calib_rs = np.empty(0)
        calib_phis = np.empty(0)
        for i in range(len(self.fns)):
            img_segs = [s for s in segments if s.imgNum == i]
            calib_seg = max(img_segs, key=operator.attrgetter('bandNum'))
            calib_rs = np.append(calib_rs, calib_seg.rs)
            calib_phis = np.append(calib_phis, calib_seg.phis)
        calib_size_px = np.mean(calib_rs)

        # Dimensions of the calibration piece
        calib_size_mm = self.env.calib_size_mm  # Outer radius of calibration piece
        tolerance = 1.1
        calib_width_mm = self.env.calib_width_mm * tolerance  # Width of the calibration piece
        pitch_mm = self.env.pitch_mm  # Nominal electrode pitch

        scale = calib_size_mm / calib_size_px

        print('Resolution: ' + str(round(scale * 1000, 2)) + ' um/px')

        calibrationCutoff = (calib_size_mm - calib_width_mm) / scale
        pitch = pitch_mm / scale

        segments = [s for s in segments if np.min(s.rs) < calibrationCutoff]

        connected_ids = []
        segments_sorted = sorted(segments, key=operator.attrgetter('comP'))
        left_starts = sorted([s for s in segments if s.imgNum == 0], key=operator.attrgetter('comR'))
        chirality = 1
        bands = []
        for start in left_starts:
            combined = [start]
            connected_ids.append(start.identity)
            # for i in range(len(self.fns) - 1):
            while True:
                # candidates are all next images
                candidates = [s for s in segments_sorted if s.imgNum ==
                              combined[-1].imgNum + 1 and s.identity not in connected_ids]
                if len(candidates) == 0:
                    break
                # Connect EP to candiate LP
                ep = combined[-1].ep
                candidates_lp = [c.lp for c in candidates]
                # sortedPt = sorted(candidates_lp, key=lambda x: vectools.pointdistPolar(x, ep))
                nearestPt = min(candidates_lp, key=lambda x: abs(x[1] - ep[1]))
                if abs(nearestPt[1] - ep[1]) > 0.25 * pitch:
                    print('breaking left')
                    break
                nearest_idx = candidates_lp.index(nearestPt)
                nearest_seg = candidates[nearest_idx]
                combined.append(nearest_seg)
                connected_ids.append(nearest_seg.identity)
            bandR = np.empty(0)
            bandP = np.empty(0)
            for c in combined:
                bandR = np.append(bandR, c.rs)
                bandP = np.append(bandP, c.phis)
            order = np.argsort(bandP)
            bandR = bandR[order]
            bandP = bandP[order]
            bands.append([bandP, bandR])
        right_starts = sorted([s for s in segments if s.imgNum == len(self.fns) -
                               1 and s.identity not in connected_ids], key=operator.attrgetter('comR'))
        for start in right_starts:
            combined = [start]
            connected_ids.append(start.identity)
            # for i in range(len(self.fns) - 1):
            while True:
                # candidates are all next images
                candidates = [s for s in segments_sorted if s.imgNum ==
                              combined[-1].imgNum - 1 and s.identity not in connected_ids]
                if len(candidates) == 0:
                    break
                # Connect EP to candiate LP
                lp = combined[-1].lp
                candidates_ep = [c.ep for c in candidates]
                nearestPt = min(candidates_ep, key=lambda x: abs(x[1] - lp[1]))
                if abs(nearestPt[1] - lp[1]) > 0.25 * pitch:
                    print('breaking')
                    break
                nearest_idx = candidates_ep.index(nearestPt)
                nearest_seg = candidates[nearest_idx]
                combined.append(nearest_seg)
                connected_ids.append(nearest_seg.identity)

            bandR = np.empty(0)
            bandP = np.empty(0)
            for c in combined:
                bandR = np.append(bandR, c.rs)
                bandP = np.append(bandP, c.phis)
            order = np.argsort(bandP)
            bandR = bandR[order]
            bandP = bandP[order]
            bands.append([bandP, bandR])
        if len(segments) != len(connected_ids):
            print(len(segments))
            print(len(connected_ids))
            #  raise Exception('Disconnected segments')

        avgR = [np.mean(b[1]) for b in bands]
        order = np.argsort(avgR)
        bands = np.array(bands)[order]

        compP = np.empty(0)
        compR = np.empty(0)
        compX = np.empty(0)
        compY = np.empty(0)
        for i, b in enumerate(bands):
            compR = np.append(compR, b[1])
            phi = b[0] + i * chirality * 2 * np.pi
            compP = np.append(compP, phi)
        order = np.argsort(compP)
        compP = compP[order]

        deltas = [abs(r - compR[i + 1]) for i, r in enumerate(compR[:-1])]
        if max(deltas) > 0.8 * pitch:
            compP = np.empty(0)
            compR = np.empty(0)
            chirality *= -1
            for i, b in enumerate(bands):
                compR = np.append(compR, b[1])
                phi = b[0] + i * chirality * 2 * np.pi
                compP = np.append(compP, phi)
            order = np.argsort(compP)
            compP = compP[order]

        compR = compR[order] * scale

        deltas = [abs(r - compR[i + 1]) for i, r in enumerate(compR[:-1])]
        print(max(deltas))
        compX = compR * np.cos(compP)
        compY = compR * np.sin(compP)

        if plot:
            figPolar = plt.figure()
            figCart = plt.figure()
            axPolar = figPolar.add_subplot(111)
            axCart = figCart.add_subplot(111)
            axPolar.plot(compP, compR, lw=0.5, c='black')
            # axPolar.plot(deltas)
            axPolar.axhline(y=calibrationCutoff * scale, lw=0.8, ls='-.', c='red')
            axPolar.axhline(y=calib_size_px * scale, lw=0.8, c='red')
            axPolar.fill_between([0, compP[-1]], calibrationCutoff * scale,
                                 calib_size_px * scale, facecolor='red', alpha=0.1)

            axCart.plot(compX, compY, lw=0.5, c='black')
            cutoffCircle = plt.Circle((0, 0), calibrationCutoff * scale,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=0.8,
                                      ls='-.')
            calibrationCircle = plt.Circle((0, 0), calib_size_mm,
                                           edgecolor='red',
                                           facecolor='none',
                                           lw=0.8,)
            wedge = Wedge((0, 0), calib_size_mm, 0, 360, width=calib_width_mm, color='red', alpha=0.1)
            axCart.add_artist(cutoffCircle)
            axCart.add_artist(calibrationCircle)
            axCart.add_artist(wedge)
            axCart.scatter(0, 0, marker='+', lw=0.5, color='red')

            axPolar.set_xlabel('Phi [rad]')
            axPolar.set_ylabel('Radius [mm]')

            axCart.set_xlabel('X [mm]')
            axCart.set_ylabel('Y [mm]')
            rmax = calib_size_mm * 1.1
            axCart.set_xlim([-rmax, rmax])
            axCart.set_ylim([-rmax, rmax])
            axCart.set_aspect('equal')

            axPolar.set_title('Reconstructed Spiral')
            axCart.set_title('Reconstructed Spiral')

            figPolar.savefig(__location__ + '/../img/out/combinedPolar.png', dpi=300)
            figCart.savefig(__location__ + '/../img/out/combinedCart.png', dpi=300)

        parametrized = (compP, compR, scale, chirality)
        return parametrized

    def loadSegments(self, fn='stitched.npy'):
        fn = __location__ + '/../data/' + fn
        segments = np.load(fn)
        return segments

    def pickleSave(self, fn='stitcher.pkl'):
        """
        Save the class with the processed images to a pickled binary object

        Parameters
        ----------
        fn: string
            filename of the output
        """
        p = pickler.Pickler()
        p.save(self, fn)

    def singleRoutine(self, fn, calibration):
        """
        Image processing routine to be parallelized

        Parameters
        ----------
        fn: string
            filename of the single image
        """
        # npzfn = 'data/' + (fn.split('/')[-1].split('.')[0]) + '.npz'
        im = singleImage.SingleImage(fn, calibration)
        # im.getFeatures(npz=npzfn)
        im.getFeatures()
        # im.setFeatures(npz=npzfn)
        im.getLines()
        return im
