import operator
import h5py
import datetime
from electrode import Electrode

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
                chirality = (i * 2) - 1
                scale = g.attrs['Scale']
                payload = (phis, rs, scale, chirality)
                self.electrodes.append(Electrode(self.serial, payload, calibration))
