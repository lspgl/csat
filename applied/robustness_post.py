import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

import matplotlib.pyplot as plt
import h5py
from electrode import Electrode
import numpy as np
import scipy.stats as st


class TestSet:

    def __init__(self):
        self.testSets = []
        self.loss = []
        self.nSets = 3

    def load(self, fn, corrections=False):
        for i in range(0, self.nSets):
            electrodes = []
            with h5py.File(__location__ + '/data/CSAT Robustness/Batch 2/' + fn + str(i + 1) + '.h5', 'r') as f:
                attributes = f.attrs
                for attr in attributes:
                    setattr(self, attr, attributes[attr])
                nGroups = 50
                for i in range(nGroups):
                    gname = 'ITER_' + str(i + 1)
                    try:
                        g = f[gname]
                    except KeyError:
                        self.loss.append(nGroups - i)
                        break
                    phis = g['spiral'][0]
                    rs = g['spiral'][1]
                    calibration = g['calibration'][:]
                    chirality = -((i * 2) - 1)
                    scale = g.attrs['Scale']
                    payload = (phis, rs, scale, chirality)
                    electrodes.append(Electrode(self.serial, payload, calibration))
            self.testSets.append(electrodes)
        print('Failed Measurements:', sum(self.loss), 'out of', self.nSets * nGroups)

    def makeKDE(self, ax, data, data_flat, title):
        t = data_flat[:, 0]
        r = data_flat[:, 1]
        tstd = np.std(t)
        rstd = np.std(r)

        rmin = np.min(r) - 2 * rstd
        tmin = np.min(t) - 2 * tstd

        rmax = np.max(r) + 2 * rstd
        tmax = np.max(t) + 2 * tstd

        # Peform the kernel density estimate
        tt, rr = np.mgrid[tmin:tmax:100j, rmin:rmax:100j]
        positions = np.vstack([tt.ravel(), rr.ravel()])
        values = np.vstack([t, r])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, tt.shape)

        ax.set_xlim(tmin, tmax)
        ax.set_ylim(rmin, rmax)

        ax.contourf(tt, rr, f, 100, cmap=plt.cm.viridis)

        ax.set_aspect('auto')
        markers = ['x', 'o', '+']
        edgecol = ['none', 'red', 'red']
        facecol = ['red', 'none', 'red']

        ax.set_title(title)
        for i, t in enumerate(data):
            marker = markers[i]
            ec = edgecol[i]
            fc = facecol[i]
            for p in t:
                ax.scatter(*p, edgecolor=ec, facecolor=fc, marker=marker, s=4, lw=0.5)

    def bandDeltas(self, ax):
        for i, t in enumerate(self.testSets):
            max_theta = []

            for e in t:
                t0 = abs(e.phis[0])
                tf = abs(e.phis[-1])
                max_theta.append(max(t0, tf))
            max_valid_theta = min(max_theta)
            max_valid_theta -= 1e-5
            space = np.linspace(0, max_valid_theta, 1000)
            equal_rs = []
            for e in t:
                max_valid_idx = np.argmax(e.phis > max_valid_theta)
                if max_valid_idx == 0:
                    max_valid_idx = np.argmax(-1 * e.phis[::-1] >= max_valid_theta)
                    fp = e.rs[::-1][:max_valid_idx]
                    xp = -1 * e.phis[::-1][:max_valid_idx]
                else:
                    fp = e.rs[:max_valid_idx]
                    xp = e.phis[:max_valid_idx]
                equal_rs.append(np.interp(space, xp, fp))
            deviations = np.std(equal_rs, axis=0)
            ax.plot(space, deviations, color=plt.cm.viridis((i) / (1.2 * (len(self.testSets) - 1))))

        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('Radius STD [mm]')
        ax.set_title('Radial Variance - Complete', loc='left')
        # ax.set_aspect('auto')

    def angularAnalysis(self):
        data = []
        data_flat = []
        start_radii = []
        start_radii_flat = []
        for t in self.testSets:
            subset = []
            subset_start = []
            for e in t:
                t0 = abs(e.phis[0])
                tf = abs(e.phis[-1])
                dt = max(t0, tf)
                r0 = abs(e.rs[0])
                rf = abs(e.rs[-1])
                dr = max(r0, rf)

                start_radii_flat.append(min(r0, rf))
                subset_start.append(min(r0, rf))

                point = [dt, dr]
                subset.append(point)
                data_flat.append(point)
            data.append(subset)
            start_radii.append(subset_start)
        data = np.array(data)
        data_flat = np.array(data_flat)

        start_grid = np.linspace(np.min(start_radii_flat) - np.std(start_radii_flat),
                                 np.max(start_radii_flat) + np.std(start_radii_flat),
                                 num=100)
        kde_start = st.gaussian_kde(start_radii_flat)

        fig = plt.figure(figsize=(16, 9))
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax1.set_xlabel('Radius [mm]')
        ax1.set_ylabel('PDF')
        ax2.set_xlabel('Angle [rad]')
        ax2.set_ylabel('Radius [mm]')

        self.makeKDE(ax2, data, data_flat, 'Band endpoint variance')
        self.bandDeltas(ax3)

        ax1.plot(start_grid, kde_start(start_grid), ls='-.', color='black')
        ax1.set_title('Radial Variance - Start', loc='left')
        for i, s in enumerate(start_radii):
            kde_loc = st.gaussian_kde(s)
            ax1.plot(start_grid, kde_loc(start_grid), color=plt.cm.viridis((i) / (1.2 * (len(start_radii) - 1))))

        fig.savefig(__location__ + '/data/CSAT Robustness/BandEndpointVariance.png', dpi=300)


def main():
    ts = TestSet()
    ts.load('CSAT_ROBUSTNESS_R')
    ts.angularAnalysis()


if __name__ == '__main__':
    main()
