import numpy as np
import cv2
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
import homography_gpu


def getHomography(fnpair, scaling=0.5, plot=False):

    fn1, fn2 = fnpair
    MIN_MATCH_COUNT = 10

    print('Loading and resizing Images')

    img1 = cv2.imread(fn1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(fn2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (0, 0), fx=scaling, fy=scaling)
    img2 = cv2.resize(img2, (0, 0), fx=scaling, fy=scaling)

    # Initiate SIFT detector
    print('SIFT Features')
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=200)
    print('Matching Features')
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print(matches)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print('Finding Homography')
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    print('Plotting')
    if plot:

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=(0, 0, 128),
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        ax.imshow(img3)
        fig.savefig('img/out/homography.png', dpi=300)

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
    h = getHomography(fnpair)
    mp = solveRotation(h)
    return mp


def getOscillation(directory, n=16):
    fns = [directory + '/cpt' + str(i + 1) + '.jpg' for i in range(n)]
    scaling = 0.25
    fnpairs = []
    for i, fn in enumerate(fns):
        try:
            fnpairs.append([fn, fns[i + 1]])
        except IndexError:
            fnpairs.append([fn, fns[0]])

    ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)
    ms = pool.map(calculateMidpoint, fnpairs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*zip(*ms))
    fig.savefig('img/out/homography_oscillation.png', dpi=300)

    print(ms)


if __name__ == '__main__':
    fn1 = 'img/src/copper/cpt3.jpg'
    fn2 = 'img/src/copper/cpt4.jpg'
    getOscillation('img/src/copper')
    #scaling = 0.25
    #h = getHomography(fn1, fn2, scaling=scaling, plot=False)
    #solveRotation(h, scaling)
