import operator
import h5py
import datetime
from electrode import Electrode

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt


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
            self.load(*args)

    def initializeFromMeasurement(self, env, electrodes, serial):
        self.env = env
        self.electrodes = sorted(electrodes, key=operator.attrgetter('chirality'))
        if self.electrodes[0].chirality == self.electrodes[1].chirality:
            raise Exception('Chirality Error')
        self.serial = serial
        t = datetime.datetime.now()
        self.timestamp = (str(t.year) + '-' + str(t.month) + '-' + str(t.day) + '-' +
                          str(t.hour) + '-' + str(t.minute) + '-' + str(t.second))

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

    def load(self, fn):
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

    def optimizeLinearity(self, data, coverage=0.9):
        opt = optimize.least_squares(self.linearity, (0, 0), args=(data, coverage), jac='2-point')
        print(opt.x)
        dx, dy = opt.x
        phi, r = data
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xnew = x + dx
        ynew = y + dy

        rn = np.sqrt(np.square(xnew) + np.square(ynew))
        phin = np.arctan2(ynew, xnew)
        phin -= phin[0]
        deltas = np.abs(np.array([phii - phin[i + 1] for i, phii in enumerate(phin[:-1])]))
        indices = np.argwhere(deltas >= np.pi).flatten()
        shift = np.zeros(len(phin))
        for idx in indices:
            inew = idx + 1
            shift[inew:] = shift[inew:] + (2 * np.pi)
        phin += shift

        dnew = (phin, rn)

        return dnew

    def linearity(self, shift, data, coverage):
        phi, r = data
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        dx, dy = shift

        xn = x + dx
        yn = y + dy

        rn = np.sqrt(np.square(xn) + np.square(yn))
        phin = np.arctan2(yn, xn)
        phin -= phin[0]
        deltas = np.abs(np.array([phii - phin[i + 1] for i, phii in enumerate(phin[:-1])]))
        indices = np.argwhere(deltas >= np.pi).flatten()
        shift = np.zeros(len(phin))
        for idx in indices:
            inew = idx + 1
            shift[inew:] = shift[inew:] + (2 * np.pi)
        phin += shift
        pkg = np.polyfit(x=phin[:int(len(phin) * coverage)], y=rn[:int(len(rn) * coverage)], deg=1, full=True)
        residual = pkg[1][0]
        return residual

    def computeGap(self):
        lengths = []
        max_phis = []
        for e in self.electrodes:
            phiAbs = np.abs(e.phis)[::e.chirality]
            npts = len(phiAbs)
            lengths.append(npts)
            pmax = phiAbs[-1]
            max_phis.append(pmax)
        max_length = min(lengths)
        max_phi = min(max_phis)

        global_phis = np.linspace(0, max_phi, num=max_length, endpoint=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, e in enumerate(self.electrodes):
            r_interp = np.interp(global_phis, np.abs(e.phis)[::e.chirality], e.rs[::e.chirality])
            data = (global_phis, r_interp)
            pnew, rnew = self.optimizeLinearity(data)
            pnew += i * np.pi
            x = rnew * np.cos(pnew)
            y = rnew * np.sin(pnew)
            ax.plot(x, y, lw=1)

        ax.set_aspect('equal')
        #ax.set_xlim([0, 10])
        #ax.set_ylim([0, 10])

        fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '_interpolated.png', dpi=300)

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]

        for i, ax in enumerate(axes):
            ax.plot(self.electrodes[i].phis, self.electrodes[i].rs)

        fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '.png', dpi=300)
