import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

import matplotlib.pyplot as plt
import h5py
from electrode import Electrode


class TestSet:

    def __init__(self):
        self.electrodes = []

    def load(self, fn, corrections=False):
        with h5py.File(__location__ + '/data/CSAT Robustness/Batch 1/' + fn, 'r') as f:
            attributes = f.attrs
            for attr in attributes:
                setattr(self, attr, attributes[attr])
            nGroups = 50
            for i in range(nGroups):
                gname = 'ITER_' + str(i + 1)
                try:
                    g = f[gname]
                except KeyError:
                    self.loss = nGroups - i
                    break
                phis = g['spiral'][0]
                rs = g['spiral'][1]
                calibration = g['calibration'][:]
                chirality = -((i * 2) - 1)
                scale = g.attrs['Scale']
                payload = (phis, rs, scale, chirality)
                self.electrodes.append(Electrode(self.serial, payload, calibration))

    def angularAnalysis(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for e in self.electrodes:
            t0 = abs(e.phis[0])
            tf = abs(e.phis[-1])
            dt = max(t0, tf)

            r0 = abs(e.rs[0])
            rf = abs(e.rs[-1])
            dr = max(r0, rf)
            ax.scatter(dt, dr, color='red', marker='x')

            print(dt, dr)
        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('Radius [mm]')

        ax.set_title('Band endpoint variance')
        fig.savefig(__location__ + '/data/CSAT Robustness/BandEndpointVariance.png', dpi=300)


def main():
    ts = TestSet()
    ts.load('CSAT_ROBUSTNESS_L1.h5')
    ts.angularAnalysis()


if __name__ == '__main__':
    main()
