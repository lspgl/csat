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

import multiprocessing as mp

import cv2

import sys
import os
import time

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
        lock = mp.Lock()

        if self.mpflag:
            self.images = Parmap(self.singleRoutine, self.fns, self.calibration, self.env, lock)
            # self.images = Parmap(self.singleRoutine, srcs, self.calibration, lock)
        else:
            for fn in self.fns:
                im = singleImage.SingleImage(fn, self.calibration)
                im.getFeatures()
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

        ampstart = 0
        ampend = 0
        for i, image in enumerate(self.images[::-1]):
            if image.start[1] > ampstart:
                pstart = image.angles[image.start[0][0]] + (i * 2 * np.pi / len(self.fns))
                rstart = image.radii[image.start[0][1]]
                ampstart = image.start[1]
                idstart = i + 1
            if image.end[1] > ampend:
                pend = image.angles[image.end[0][0]] + (i * 2 * np.pi / len(self.fns))
                rend = image.radii[image.end[0][1]]
                ampend = image.end[1]
                idend = i + 1

            for j, coord in enumerate(zip(image.r, image.phi)):
                rs, phis = coord
                phis = image.angles[np.array(phis)] + (i * 2 * np.pi / len(self.fns))
                rs = image.radii[np.array(rs)]
                dr = rs[:-1] - rs[1:]
                # std = np.std(dr)
                outlier = np.max(np.abs(dr))
                if outlier < 500:
                    s = (phis, rs, i, j)
                    segments.append(s)
                    if plot:
                        ax.plot(phis, rs, lw=0.2)

        self.startAngle = pstart
        self.startRadius = rstart
        self.idstart = idstart

        self.endAngle = pend
        self.endRadius = rend
        self.idend = idend

        if plot:
            # ax.set_aspect('equal')
            ax.set_xlabel('Angle [rad]')
            ax.set_ylabel('Radius [px]')
            # ax.set_ylim([1300, 1550])
            fig.savefig(__location__ + '/../img/out/stitched.png', dpi=600)

        if tofile:
            np.save(__location__ + '/../data/stitched.npy', segments)

        return segments

    def getNearestSegment(self, refPoint, segments, fraction=1):
        r0 = refPoint[1]
        dr_min = np.array([np.min(np.abs(s.rs[:int(fraction * (len(s.rs) - 1))] - r0)) for s in segments])
        return segments[np.argmin(dr_min)], np.min(dr_min)

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
        plot = True
        segments = [Segment(*coords, identity=i) for i, coords in enumerate(segments)]
        # Calibration piece
        calib_size_px = np.mean(np.array([x[2] for x in self.calibration]))

        # Dimensions of the calibration piece
        calib_size_mm = self.env.calib_size_mm  # Outer radius of calibration piece
        tolerance = 1.1
        calib_width_mm = self.env.calib_width_mm * tolerance  # Width of the calibration piece
        pitch_mm = self.env.pitch_mm  # Nominal electrode pitch

        scale = calib_size_mm / calib_size_px

        print(_C.BOLD + _C.CYAN + 'Resolution: ' + str(round(scale * 1000, 2)) + ' um/px' + _C.ENDC)

        calibrationCutoff = (calib_size_mm - calib_width_mm) / scale
        pitch = pitch_mm / scale

        segments = [s for s in segments if np.min(s.rs) < calibrationCutoff]

        connected_ids = []
        segments_sorted = sorted(segments, key=operator.attrgetter('comP'))
        left_starts = sorted([s for s in segments if s.imgNum == 0], key=operator.attrgetter('comR'))
        chirality = 1
        bands = []
        Lbreak = 0

        for start in left_starts:
            skipFlag = False
            combined = [start]
            connected_ids.append(start.identity)
            # for i in range(len(self.fns) - 1):
            while True:
                # candidates are all next images
                candidates = [s for s in segments_sorted if s.imgNum ==
                              combined[-1].imgNum + 1 and s.identity not in connected_ids]
                if len(candidates) == 0:
                    break
                # Connect EP to candiate
                ep = combined[-1].ep
                nearest_seg, dr = self.getNearestSegment(ep, candidates)

                if dr > 3.0:
                    # if dr > 10.0 and (dr > 100 and ep[1] > 3000):
                    print(_C.BLUE + 'Breaking left' + _C.ENDC)
                    Lbreak += 1
                    if Lbreak > 1:
                        pass
                        # skipFlag = True
                    break

                combined.append(nearest_seg)
                connected_ids.append(nearest_seg.identity)
            bandR = np.empty(0)
            bandP = np.empty(0)
            if not skipFlag:
                for c in combined:
                    bandR = np.append(bandR, c.rs)
                    bandP = np.append(bandP, c.phis)
                order = np.argsort(bandP)
                bandR = bandR[order]
                bandP = bandP[order]
                bands.append([bandP, bandR])
        right_starts = sorted([s for s in segments if s.imgNum == len(self.fns) -
                               1 and s.identity not in connected_ids], key=operator.attrgetter('comR'))
        Rbreak = 0
        for start in right_starts:
            skipFlag = False
            combined = [start]
            connected_ids.append(start.identity)
            # for i in range(len(self.fns) - 1):
            while True:
                # candidates are all next images
                candidates = [s for s in segments_sorted if s.imgNum ==
                              combined[-1].imgNum - 1 and s.identity not in connected_ids]
                if len(candidates) == 0:
                    break
                # Connect LP to candiate
                lp = combined[-1].lp
                nearest_seg, dr = self.getNearestSegment(lp, candidates)
                if dr > 3.0:
                    print(_C.BLUE + 'Breaking right' + _C.ENDC)
                    Rbreak += 1
                    if Rbreak > 1:
                        pass
                        # skipFlag = True
                    break

                combined.append(nearest_seg)
                connected_ids.append(nearest_seg.identity)

            bandR = np.empty(0)
            bandP = np.empty(0)
            if not skipFlag:
                for c in combined:
                    bandR = np.append(bandR, c.rs)
                    bandP = np.append(bandP, c.phis)
                order = np.argsort(bandP)
                bandR = bandR[order]
                bandP = bandP[order]
                bands.append([bandP, bandR])
        if len(segments) != len(connected_ids):
            print(_C.RED + str(len(segments) - len(connected_ids)) + ' disconncted segments (' +
                  str(len(connected_ids)) + '/' + str(len(segments)) + ')' + _C.ENDC)
            #  raise Exception('Disconnected segments')

        avgR = [np.mean(b[1]) for b in bands]
        order = np.argsort(avgR)
        bands = np.array(bands)[order]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        band_segments = [Segment(b[0], b[1], imgNum=0, bandNum=0, identity=i) for i, b in enumerate(bands)]

        band_groups = []
        for b in band_segments:
            ep = b.ep
            others = [s for s in band_segments if s.identity != b.identity]
            nearest_band, dr = self.getNearestSegment(b.ep, others, fraction=(1 / len(self.fns)))
            if dr < 3.0:
                contained = False
                for g in band_groups:
                    if b.identity in g:
                        g.append(nearest_band.identity)
                        contained = True
                        break
                    elif nearest_band.identity in g:
                        g.append(b.identity)
                        contained = True
                        break
                if not contained:
                    band_groups.append([b.identity, nearest_band.identity])
            else:
                band_groups.append([b.identity])
            ax.plot(b.phis, b.rs, lw=0.2)
        max_group = max(band_groups, key=len)
        max_bands = [b for b in band_segments if b.identity in max_group]
        for b in max_bands:
            # ax.plot(b.phis, b.rs, lw=0.2)
            pass
        fig.savefig('bands_debug.png', dpi=300)

        compP = np.empty(0)
        compR = np.empty(0)
        for i, b in enumerate(max_bands):
            compR = np.append(compR, b.rs)
            phi = b.phis + i * chirality * 2 * np.pi
            compP = np.append(compP, phi)
        order = np.argsort(compP)
        compP = compP[order]

        deltas = [abs(r - compR[i + 1]) for i, r in enumerate(compR[:-1])]
        if max(deltas) > 0.8 * pitch:
            compP = np.empty(0)
            compR = np.empty(0)
            chirality *= -1
            for i, b in enumerate(max_bands):
                compR = np.append(compR, b.rs)
                phi = b.phis + i * chirality * 2 * np.pi
                compP = np.append(compP, phi)
            order = np.argsort(compP)
            compP = compP[order]
        # scale = 1
        compR = compR[order] * scale
        compP = chirality * compP[::chirality]

        self.startAngle = chirality * self.startAngle

        self.endAngle = (chirality * -1 + 1) * np.pi + (chirality * self.endAngle)

        opening_angle = abs(self.startAngle - self.endAngle)
        print('OPENING ANGLE:', opening_angle)
        # print(self.startAngle)

        while True:
            self.endAngle += 2 * np.pi
            test_idx = np.argmax(compP > self.endAngle)
            if test_idx == 0:
                self.endAngle -= 2 * np.pi
                end_idx = np.argmax(compP > self.endAngle)
                break
        loss = len(compP) - end_idx

        if loss > 1000:
            #print('Cutoff Loss Overflow:', loss)
            raise Exception('Cutoff Loss Overflow ' + str(loss))
            #end_idx = len(compP) - 1

        start_idx = np.argmax(compP > self.startAngle)
        if start_idx == 0:
            print('Unraveling Start Angle')
            self.startAngle += 2 * np.pi
            start_idx = np.argmax(compP > self.startAngle)
        # start_idx = 0
        # print(compP)

        dr_start = abs(compR[::chirality][start_idx] / scale - self.startRadius)
        dr_end = abs(compR[::chirality][end_idx] / scale - self.endRadius)

        if max(dr_start, dr_end) > 0.5 * pitch:
            print('!!!!!!!!!!!!!!!!!!')
            print('Band Start Metric:')
            print('r:', self.startRadius)
            print('t:', self.startAngle % (2 * np.pi))
            print('dr:', dr_start)
            print('id:', self.idstart)
            print('------------------')
            print('Band End Metric:')
            print('r:', self.endRadius)
            print('t:', self.endAngle % (2 * np.pi))
            print('dr:', dr_end)
            print('id:', self.idend)
            print('------------------')
            print('Pitch Threshold (x0.5):', pitch)
            print('!!!!!!!!!!!!!!!!!!')
            raise Exception('Band Cutoff Mismatch')

        compP = compP[start_idx:end_idx]
        compP -= compP[0]
        compP = chirality * compP[::chirality]
        compR = compR[::chirality][start_idx:end_idx][::chirality]

        compX = compR * np.cos(compP)
        compY = compR * np.sin(compP)
        if plot:
            figPolar = plt.figure()
            figCart = plt.figure()
            axPolar = figPolar.add_subplot(111)
            axCart = figCart.add_subplot(111)
            axPolar.plot(compP, compR, lw=0.5, c='black')
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

    def singleRoutine(self, fn, calibration, env, lock):
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
        im.getFeatures(env=env, lock=lock)
        # im.setFeatures(npz=npzfn)
        im.getLines()
        return im
