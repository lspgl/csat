import silx
from PIL import Image
from silx.test.utils import utilstest
from silx.image import sift
import numpy as np
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp


def getHomography_GPU(fns, plot=False):
    path1, path2 = fns
    pil1 = Image.open(path1)
    pil2 = Image.open(path2)
    image1 = np.asarray(pil1)
    image2 = np.asarray(pil2)
    print('Running SIFT on GPU')
    sift_ocl = sift.SiftPlan(image1.shape, image1.dtype, template=image1, devicetype='GPU')
    kp1 = sift_ocl(image1)
    kp2 = sift_ocl(image2)

    print('Running Match on GPU')
    matchplan = sift.MatchPlan(devicetype='GPU')
    match = matchplan(kp1, kp2, raw_results=True)

    if len(match) == 0:
        return None

    src_pts = np.empty((len(match), 2))
    dst_pts = np.empty((len(match), 2))
    for i, m in enumerate(match):
        src_pt = np.array((kp1[m[0]].x, kp1[m[0]].y))
        dst_pt = np.array((kp2[m[1]].x, kp2[m[1]].y))
        src_pts[i] = src_pt
        dst_pts[i] = dst_pt
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if plot:
        h, w, c = image1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.imshow(image2)
        fig.savefig('img/out/homography_png.png', dpi=300)

    return M


def solveRotation(h, scaling=1.0):
    costheta = np.mean([h.item(0, 0), h.item(1, 1)])
    sintheta = np.mean([h.item(1, 0), -h.item(0, 1)])
    tx = h.item(0, 2)
    ty = h.item(1, 2)
    b = np.array([-tx, -ty])
    a = np.array([[costheta - 1, -sintheta], [sintheta, costheta - 1]])

    x = np.linalg.solve(a, b) / scaling
    return x


def calculateMidpoint(fnpair):
    h = getHomography_GPU(fnpair)
    if h is not None:
        midpoint = solveRotation(h)
        return midpoint
    else:
        return None


def getOscillation(directory, n=16):
    fns = [directory + '/cpt' + str(i + 1) + '.jpg' for i in range(n)]
    fnpairs = []
    for i, fn in enumerate(fns):
        try:
            fnpairs.append([fn, fns[i + 1]])
        except IndexError:
            fnpairs.append([fn, fns[0]])

    #ncpus = mp.cpu_count()
    #pool = mp.Pool(ncpus)
    #ms = pool.map(calculateMidpoint, fnpairs)

    ms = [calculateMidpoint(fnpair) for fnpair in fnpairs]

    faulty = [i + 1 for i, m in enumerate(ms) if m is None]

    ms_fixed = [m for m in ms if m is not None]

    midpoints = filterOutliers(ms_fixed)

    print(ms)
    print(faulty)
    print(ms_fixed)

    av_midpoint = np.mean(midpoints, axis=0)
    print(av_midpoint)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(*zip(*ms_fixed), color='r')
    ax.plot(*zip(*midpoints), lw=0.1, color='g')
    ax.scatter(*av_midpoint, color='b')
    fig.savefig('img/out/homography_oscillation_GPU.png', dpi=300)


def filterOutliers(pts):
    avg_raw = np.mean(pts, axis=0)
    distances = [np.linalg.norm(pt - avg_raw) for pt in pts]
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    m = 1
    new_pts = [pt for i, pt in enumerate(pts) if abs(distances[i] - avg_dist) < m * std_dist]
    return new_pts


if __name__ == '__main__':
    #fns = ['img/src/copper/cpt1.jpg', 'img/src/copper/cpt2.jpg']
    #h = getHomography_GPU(fns)
    # print(h)
    directory = 'img/src/copper'
    directory = '../hardware'
    getOscillation(directory, n=22)
