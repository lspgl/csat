import operator
import h5py
import datetime
from electrode import Electrode

import time

import numpy as np
from scipy import optimize
from scipy import ndimage

import matplotlib.pyplot as plt
from tools.colors import Colors as _C

import os
import sys
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__ + '/../')


class Pair:

    def __init__(self, *args, fromFile=False, **kwargs):
        if not fromFile:
            self.initializeFromMeasurement(*args, **kwargs)
        else:
            self.load(*args, **kwargs)

    def initializeFromMeasurement(self, env, electrodes, serial, corrections=False):
        self.serial = serial
        self.env = env
        self.electrodes = sorted(electrodes, key=operator.attrgetter('chirality'))
        if self.electrodes[0].chirality == self.electrodes[1].chirality:
            raise Exception('Chirality Error')
        if corrections:
            self.correctMidpoint()
            self.optimizeRotation()
        t = datetime.datetime.now()
        self.timestamp = (str(t.year) + '-' + str(t.month) + '-' + str(t.day) + '-' +
                          str(t.hour) + '-' + str(t.minute) + '-' + str(t.second))

    def correctMidpoint(self):
        print(_C.BLUE + 'Correcting midpoint' + _C.ENDC)
        lengths = []
        max_phis = []
        for i, e in enumerate(self.electrodes[:]):
            data = (np.abs(e.phis[::e.chirality]), e.rs[::e.chirality])
            popt, ropt = self.optimizeLinearity(data)
            pnew, rnew = self.smoothing((popt, ropt))
            phiAbs = np.abs(pnew)
            npts = len(phiAbs)
            lengths.append(npts)
            pmax = phiAbs[-1]
            max_phis.append(pmax)

            self.electrodes[i].phis = pnew[:]
            self.electrodes[i].rs = rnew[:]

        max_length = min(lengths)
        max_phi = min(max_phis)

        global_phis = np.linspace(0, max_phi, num=max_length, endpoint=True)

        for i, e in enumerate(self.electrodes[:]):
            r_interp = np.interp(global_phis, e.phis, e.rs)
            self.electrodes[i].rs = r_interp[:]
            self.electrodes[i].phis = global_phis[:] + i * np.pi

    def store(self, fn=None):
        if fn is not None:
            self.serial = fn
        attributes = {'serial': self.serial,
                      'timestamp': self.timestamp,
                      'calib size': self.env.calib_size_mm,
                      'calib width': self.env.calib_width_mm,
                      }

        fn = 'CSAT_' + self.serial + '.h5'
        with h5py.File(__location__ + '/data/' + fn, 'w') as f:
            for key in attributes:
                f.attrs[key] = attributes[key]
            for e in self.electrodes:
                if e.chirality == 1:
                    gname = 'L-Spiral'
                else:
                    gname = 'R-Spiral'
                g = f.create_group(gname)
                g.create_dataset('spiral', data=[e.phis, e.rs])
                g.attrs['Scale'] = e.scale
                g.create_dataset('calibration', data=e.calibration)

    def load(self, fn, corrections=False):
        self.electrodes = []
        with h5py.File(__location__ + '/data/' + fn, 'r') as f:
            attributes = f.attrs
            for attr in attributes:
                setattr(self, attr, attributes[attr])
            groups = [f['L-Spiral'], f['R-Spiral']]
            for i, g in enumerate(groups):
                phis = g['spiral'][0]
                rs = g['spiral'][1]
                calibration = g['calibration'][:]
                chirality = -((i * 2) - 1)
                scale = g.attrs['Scale']
                payload = (phis, rs, scale, chirality)
                self.electrodes.append(Electrode(self.serial, payload, calibration))
        if corrections:
            self.correctMidpoint()
            self.optimizeRotation()

    def computeGaps(self, shift, opt=True, plot=False):
        e0, eRot = self.electrodes[:]
        phiRef = eRot.phis[:] + shift

        validRot_start = np.argmax(phiRef > 2 * np.pi)
        validRot_end = np.argmax(e0.phis > phiRef[-1] - 2 * np.pi)
        if validRot_end == 0:
            validRot_end = len(phiRef) - 1
        if abs(len(eRot.rs[validRot_start:]) - len(e0.rs[:validRot_end])) == 1:
            if len(eRot.rs[validRot_start:]) < len(e0.rs[:validRot_end]):
                validRot_start -= 1
            else:
                validRot_start += 1
        gapRot = eRot.rs[validRot_start:] - e0.rs[:validRot_end]

        valid0_start = np.argmax(e0.phis > phiRef[0])
        valid0_end = np.argmax(phiRef > e0.phis[-1])
        if valid0_end == 0:
            valid0_end = len(e0.phis) - 1
        if abs(len(e0.rs[valid0_start:]) - len(eRot.rs[:valid0_end])) == 1:
            if len(e0.rs[valid0_start:]) < len(eRot.rs[:valid0_end]):
                valid0_start -= 1
            else:
                valid0_start += 1
        gap0 = e0.rs[valid0_start:] - eRot.rs[:valid0_end]

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(e0.phis[:valid0_end], gap0, color=plt.cm.viridis(0), lw=0.2)
            ax.plot(phiRef[:validRot_end], gapRot, color=plt.cm.viridis(1 / 1.5), lw=0.2)
            ax.set_title('Electrode Gaps')
            ax.set_xlabel('Angle [rad]')
            ax.set_ylabel('Gap [mm]')
            fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '_gaps.png', dpi=300)
        if opt:
            retarr = np.append(np.array(gap0), np.array(gapRot))
            retval = np.std(retarr)
            return retval
        else:
            return (gap0, gapRot)

    def optimizeRotation(self, plot=False):
        print(_C.BLUE + 'Optimizing relative angle' + _C.ENDC)
        resolution = 10
        scanRange = np.linspace(-np.pi, np.pi, num=resolution)
        vfunc = np.vectorize(self.computeGaps)
        stds = vfunc(scanRange)
        minCoarse = scanRange[np.argmin(stds)]

        resolutionFine = 100
        scanRangeFine = np.linspace(minCoarse - (2 * np.pi / resolution), minCoarse +
                                    (2 * np.pi / resolution), num=resolutionFine)
        stdsFine = vfunc(scanRangeFine)
        minFine = scanRangeFine[np.argmin(stdsFine)]
        self.electrodes[1].phis += minFine
        self.computeGaps(shift=0, opt=False, plot=True)
        plot = True
        if plot:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.plot(scanRange, stds, lw=1, color=plt.cm.viridis(0.7))
            ax.axvline(x=minCoarse, ls='-.', color=plt.cm.viridis(0.7))
            ax.plot(scanRangeFine, stdsFine, lw=1, color=plt.cm.viridis(0))
            ax.axvline(x=minFine, ls='-.', color=plt.cm.viridis(0))
            ax.set_xlabel('Shift Angle [rad]')
            ax.set_ylabel('Gap STD [mm]')
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig('shiftDebug.png', dpi=300)

    def optimizeLinearity(self, data, coverage=0.9):
        opt = optimize.least_squares(self.linearity, (0, 0), args=(data, coverage), jac='2-point')
        dnew = self.translatePolar(opt.x, data)
        return dnew

    def linearity(self, translation, data, coverage):
        phin, rn = self.translatePolar(translation, data)
        pkg = np.polyfit(x=phin[:int(len(phin) * coverage)], y=rn[:int(len(rn) * coverage)], deg=1, full=True)
        residual = pkg[1][0]
        return residual

    def translatePolar(self, translation, data):
        phi, r = data
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        dx, dy = translation

        xn = x + dx
        yn = y + dy

        rn = np.sqrt(np.square(xn) + np.square(yn))
        phin = np.arctan2(yn, xn)
        phin -= phin[0]
        deltas = phin[:-1] - phin[1:]

        indices = np.argwhere(deltas >= np.pi).flatten()
        shift = np.zeros(len(phin))
        for idx in indices:
            inew = idx + 1
            shift[inew:] = shift[inew:] + (2 * np.pi)
        phin += shift

        return (phin, rn)

    def smoothing(self, data):
        phis, rs = data
        W = 10  # Window size
        nrows = rs.size - W + 1
        n = rs.strides[0]
        a2D = np.lib.stride_tricks.as_strided(rs, shape=(nrows, W), strides=(n, n))
        out = np.std(a2D, axis=1)
        cutoff = np.argmax(out > 0.015)
        if cutoff != 0:
            rbase = rs[:cutoff]
            rend = rs[cutoff:]

            delta_base = rbase[:-1] - rbase[1:]
            delta_base_std = np.std(delta_base)
            delta_base_avg = np.mean(np.abs(delta_base))

            smoothed_section = np.ones(len(rs) - cutoff) * rbase[-1]
            last_true = 0
            for i, pt in enumerate(rend[1:]):
                if abs(pt - smoothed_section[i - 1]) < delta_base_avg + 3 * delta_base_std:
                    smoothed_section[i] = pt
                    last_true = i
                else:
                    smoothed_section[i] = smoothed_section[i - 1]

            fixed_r = np.append(rbase, smoothed_section[:last_true])
            fixed_p = phis[:len(fixed_r)]

            return fixed_p, fixed_r
        else:
            print('No smoothing necessary')
            return phis, rs

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, e in enumerate(self.electrodes):
            x = e.rs * np.cos(e.phis) / e.scale
            y = e.rs * np.sin(e.phis) / e.scale
            ax.plot(x, y, color=plt.cm.viridis(i / 1.5), lw=0.8)
        ax.set_aspect('equal')
        ax.set_title('Combined Electrode')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        #ax.set_xlim([-22, 22])
        #ax.set_ylim([-22, 22])
        ax.set_xlim([-4000, 4000])
        ax.set_ylim([-4000, 4000])

        fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '.png', dpi=300)
