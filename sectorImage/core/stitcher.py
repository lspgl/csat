from . import singleImage
import numpy as np
import operator
from .toolkit import pickler
from .toolkit.parmap import Parmap
from .toolkit.colors import Colors as _C
from .toolkit import vectools
from .toolkit.segment import Segment
import matplotlib.pyplot as plt
import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Stitcher:

    def __init__(self, fns, calibration=None, mpflag=True):
        """
        Stitching class to combine multiple processed images

        Parameters
        ----------
        fns: List of strings
            filenames to be read and combined. The combination is done in the order of the supplied list
        mpflag: bool, optional
            multiprocessing flag. If on, the image processing is distributed over the available cores.
            This disables plotting of individual images. Default is True
        """
        self.fns = fns
        self.images = []
        self.mpflag = mpflag

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

    def stitchImages(self, plot=True, tofile=False):
        print(_C.MAGENTA + 'Stitching images' + _C.ENDC)
        """
        Stitch the parametrized band midpoints and plot the output

        Parameters
        ----------
        plot: bool, optional
            plot the output. Default True
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

    def combineSegments(self, segments, plot=True):
        segments = [Segment(*coords, identity=i) for i, coords in enumerate(segments)]

        # Calibrate to mm
        calib_sizes = np.array(0)
        for i in range(len(self.fns)):
            img_segs = [s for s in segments if s.imgNum == i]
            calib_seg = max(img_segs, key=operator.attrgetter('bandNum'))
            size = np.mean(calib_seg.rs)
            calib_sizes = np.append(calib_sizes, size)
        calib_size_px = np.mean(calib_sizes)
        print(calib_size_px)
        calib_size_mm = 68.0
        scale = calib_size_mm / calib_size_px

        # This should be calculated from the physical size
        calibrationCutoff = 4800
        segments = [s for s in segments if np.min(s.rs) < calibrationCutoff]

        connected_ids = []
        segments_sorted = sorted(segments, key=operator.attrgetter('comP'))
        left_starts = sorted([s for s in segments if s.imgNum == 0], key=operator.attrgetter('comR'))

        bands = []
        for start in left_starts:
            combined = [start]
            for i in range(len(self.fns) - 1):
                candidates = [s for s in segments_sorted if s.imgNum == i + 1 and s.identity not in connected_ids]
                if len(candidates) > 0:
                    ep = combined[-1].ep
                    candidates_lp = [c.lp for c in candidates]
                    nearestPt = min(candidates_lp, key=lambda x: vectools.pointdistPolar(x, ep))
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
            #ax.plot(bandP, bandR, lw=0.5)

        avgR = [np.mean(b[1]) for b in bands]
        order = np.argsort(avgR)
        bands = np.array(bands)[order]

        compP = np.array(0)
        compR = np.array(0)
        compX = np.array(0)
        compY = np.array(0)
        for i, b in enumerate(bands):
            compR = np.append(compR, b[1])
            phi = b[0] + i * 2 * np.pi
            compP = np.append(compP, phi)

        order = np.argsort(compP)
        compP = compP[order]
        compR = compR[order] * scale
        compX = compR * np.cos(compP)
        compY = compR * np.sin(compP)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(compX, compY, lw=0.5)

            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_aspect('equal')
            ax.set_title('Reconstructed Spiral')

            fig.savefig(__location__ + '/../img/out/combined.png', dpi=600)

        return (compP, compR)

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
