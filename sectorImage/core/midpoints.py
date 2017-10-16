import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


class Walker:

    def __init__(self, image):
        self.image = image
        self.skeleton = self.skeletonize(self.image)

    def walkSkeleton(self, plot=False, maxwidth=10):
        t0 = time.time()

        coords, launchpoints, endpoints, terminatinos = self.scan(
            self.skeleton, maxwidth=maxwidth)
        print('Scan time Top:', str(round(time.time() - t0, 2)), 's')

        t0 = time.time()
        t_skeleton = np.flipud(self.skeleton)
        t_coords, t_launchpoints, t_endpoints, t_terminations = self.scan(
            t_skeleton, maxwidth=maxwidth)
        print('Scan time Bottom:', str(round(time.time() - t0, 2)), 's')

        t0 = time.time()
        ud_coords = [[[self.skeleton.shape[0] - 1 - tc[0], tc[1]] for tc in t_coord] for t_coord in t_coords]
        #ud_launchpoints = [[self.skeleton.shape[0] - 1 - lp[0], lp[1]] for lp in t_launchpoints]
        ud_endpoints = [[self.skeleton.shape[0] - 1 - ep[0], ep[1]] for ep in t_endpoints]
        print('Transformation time:', str(round(time.time() - t0, 2)), 's')
        for i, p in enumerate(ud_endpoints):
            p_ext = [[p[0], p[1] + width] for width in range(-maxwidth // 2, maxwidth // 2)]
            check = True in [p in launchpoints for p in p_ext]
            if not check:
                coords.append(ud_coords[i])
                launchpoints.append(ud_endpoints[i])

        r, phi = self.splitCoords(coords)
        # TODO: Coords need to be sorted after insertion of an UD band
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for c in coords:
                ax.plot(*zip(*c), color='black', lw=0.1)
            fig.savefig('img/out/logic.png', dpi=600)

        return (r, phi)

    def splitCoords(self, coords):
        r = [[b[1] for b in band] for band in coords]
        phi = [[b[0] for b in band] for band in coords]

        return (r, phi)

    def skeletonize(self, img):
        t0 = time.time()
        # Initialize arrays to do computation in place
        skeleton = np.zeros(img.shape, np.uint8)
        eroded = np.zeros(img.shape, np.uint8)
        temp = np.zeros(img.shape, np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while(True):
            cv2.erode(img, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(img, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            img, eroded = eroded, img  # Swap instead of copy

            if cv2.countNonZero(img) == 0:
                print('Skeleton time:', str(round(time.time() - t0, 2)), 's')
                return skeleton

    def scan(self, skeleton, maxwidth=10, nbands=np.inf):
        coords = []

        head = [0, 0]
        launchpoints = []
        endpoints = []
        terminations = []
        bands_scanned = 0

        while head[1] < skeleton.shape[1] - 1:
            current = []
            if bands_scanned >= nbands:
                break
            try:
                while not skeleton[head[0], head[1]]:
                    # Advance head to first edge
                    head[1] += 1
            except IndexError:
                print('reached end of image')
                break
            current.append(head[:])
            launchpoints.append(head[:])

            termination = -1
            while head[0] < skeleton.shape[0] - 1:
                # Drop head by one line
                head[0] += 1
                scan_cell = skeleton[head[0], head[1] - maxwidth // 2: head[1] + maxwidth // 2]
                high = False
                edges = 0
                for v in scan_cell:
                    if v and not high:
                        high = True
                        edges += 1
                    if not v:
                        high = False

                if edges > 1:
                    print('line split termination at', head)
                    terminations.append(head[:])
                    break
                # Start by scanning to the right
                parity = 1
                head_ref = head[1]
                n = 1

                while not skeleton[head[0], head[1]] and n < maxwidth:
                    # Move head alternating to right and left with increasing distance
                    # until a point is found
                    head[1] = head_ref + (n * parity)
                    if parity == -1:
                        n += 1
                    parity *= -1
                if n == maxwidth:
                    head[1] = head_ref
                    termination += 1
                    if termination == 1:
                        print('line outrun termination at ', head)
                        terminations.append(head[:])
                        break
                else:
                    termination = -1
                current.append(head[:])
            coords.append(current[:])
            endpoints.append(head[:])
            head = launchpoints[-1][:]
            while skeleton[head[0], head[1]]:
                head[1] += 1
            bands_scanned += 1

        return coords, launchpoints, endpoints, terminations

    def rotatePath(self, path, lx, ly):
        # Rotate path coordinates by 180 degrees with respect to the
        # center of image with dimensions lx,ly
        pflipped = []
        for p in path:
            x, y = p
            xnew = lx - x
            pflipped.append([xnew, y])
        return pflipped
