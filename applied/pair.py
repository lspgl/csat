import operator
import h5py
import datetime
from electrode import Electrode

import numpy as np

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

    def store(self):
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
            # global_phis += i * np.pi
            shiftx = 0
            shifty = 0
            x = r_interp / e.scale * np.cos(global_phis) + i * shiftx
            y = r_interp / e.scale * np.sin(global_phis) + i * shifty
            ax.plot(x, y, lw=1)
        ax.set_aspect('equal')
        fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '_interpolated.png', dpi=300)

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]

        for i, ax in enumerate(axes):
            ax.plot(self.electrodes[i].phis, self.electrodes[i].rs)

        fig.savefig(__location__ + '/data/plots/' + str(self.serial) + '.png', dpi=300)
