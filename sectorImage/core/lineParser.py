import numpy as np
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt


def LineParser(edges, smoothing=None, plot=False):
    structure = np.ones((3, 3))
    labeled_edges, num_features = ndimage.label(edges[1:-1], structure=structure)
    print(num_features)
    unique, counts = np.unique(labeled_edges, return_counts=True)
    bands = []
    for u, c in zip(unique, counts):
        if c > 100 and u != 0:
            band = np.argwhere(labeled_edges == u)
            bands.append(band)

    print('Interpolating Fast Features')
    rsplines = []
    phis = []
    r_raw = []
    for path in bands:
        phi = [float(c[0]) for c in path]
        r = [float(c[1]) for c in path]
        # spline = scipy.interpolate.UnivariateSpline(phi, r, s=smoothing)
        phis.append(phi)
        # rs = spline(phi)
        # rsplines.append(rs)
        r_raw.append(r)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for rs, phi in zip(rsplines, phis):
            ax.plot(phi, rs, lw=0.2, color='blue')
        fig.savefig('img/out/gradients_parsed.jpg', dpi=300)

    # return radius and angle for the interpolated paths
    # return rsplines, phis
    return r_raw, phis
