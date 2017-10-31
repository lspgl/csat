import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
import time

# Walker class to follow band edges
# Edge coordinates are calculated for each band
# Calculation only sweeps along the edges without sampling
# the entire image
"""
DEPRECATED
Use Midpoints.py instead
"""


class Walker:

    def __init__(self):
        # Initialize the walker head at the origin in an inactive state
        self.active = False
        self.position = [0, 0]
        # Walker launchpoints
        self.launchpoints = [[0, 0]]
        # Walker endpoints
        self.endpoints = []

        # Does the image contain a band end
        self.containEnd = False
        # Chirality - only if there is a band end
        self.chirality = None
        # Number open lines
        self.terminations = []

    def scanEnviron(self, matrix):
        # Scan the immediate environment around the walker
        environ = {
            'd': False,
            'dl': False,
            'dr': False,
            'r': False,
            'l': False,
            'border': False,
        }
        if matrix[self.position[0], self.position[1] - 1] > 0:
            environ['l'] = True
        if matrix[self.position[0], self.position[1] + 1] > 0:
            environ['r'] = True
        # If the lower neighbors are out of bounds, an inex error is caught to
        # signify the end of the image
        try:
            if matrix[self.position[0] + 1, self.position[1]] > 0:
                environ['d'] = True
            if matrix[self.position[0] + 1, self.position[1] + 1] > 0:
                environ['dr'] = True
            if matrix[self.position[0] + 1, self.position[1] - 1] > 0:
                environ['dl'] = True
        except IndexError:
            environ['border'] = True

        return environ

    def dropStep(self, key):
        # Drop the walker to the next line with a shift given by the
        # environment key
        self.position[0] += 1
        if key == 'dl':
            self.position[1] -= 1
        elif key == 'dr':
            self.position[1] += 1

    def scan(self, matrix, hold='l', gapTolerance=50):
        # Scan along a single line, hold either to the left or right
        # edge of a face.
        # gapTolerance gives the maximum radius in which concave edges
        # are scanned. If no edges are found in this radius the band is considered to be
        # terminated.

        # Define the key order l->r or r->l depending on hold
        if hold == 'l':
            keyOrder = ['dl', 'd', 'dr']
            parity = 1
        else:
            keyOrder = ['dr', 'd', 'dl']
            parity = -1

        # Container for path coordinates
        path = []
        # While bottom of image is not reached, scan for points
        while self.position[0] < np.shape(matrix)[0]:
            # Search for activation point
            if not self.active:
                # Make sure active point hasn't been sampled before
                if matrix[self.position[0], self.position[1]] > 0 and self.position != self.launchpoints[-1]:
                    # Activate and log the starting point
                    # Point is not yet added to the container
                    self.active = True
                    self.launchpoints.append(copy.copy(self.position))

                else:
                    # Move one point to the right and try again until edge of image is reached
                    if self.position[1] < np.shape(matrix)[1] - 1:
                        self.position[1] += 1
                    else:
                        # None return serves as marker for a completed sweep without a band
                        return None
            else:
                # Initialize environment of walker
                environ = self.scanEnviron(matrix)
                # Go to border according to hold
                while environ[hold]:
                    self.position[1] -= 1 * parity
                    environ = self.scanEnviron(matrix)
                # Add point to container
                path.append(copy.copy(self.position))
                # Drop out if bottom of image is reached
                if environ['border']:
                    break

                # Assume no row is present
                rowFound = False
                # Counter for gapTolerance
                counter = 0
                while not rowFound and counter < gapTolerance:
                    # Check if a field in the environment is free
                    # ordered by hold
                    if environ[keyOrder[0]]:
                        self.dropStep(keyOrder[0])
                        rowFound = True
                    elif environ[keyOrder[1]]:
                        self.dropStep(keyOrder[1])
                        rowFound = True
                    elif environ[keyOrder[2]]:
                        self.dropStep(keyOrder[2])
                        rowFound = True
                    # If none of the immediate fields are free
                    # move to the next three fields and scan again
                    else:
                        self.position[1] += 3 * parity
                        environ = self.scanEnviron(matrix)
                        counter += 1
                # If no immediate neighbors have been found in
                # gapTolerance iterations, consider the band to be terminated
                if not rowFound:
                    print('Line Termination Detected')
                    self.terminations.append(self.position[:])
                    self.active = False
                    return path

        # Add the last point to the log
        self.endpoints.append(copy.copy(self.position))
        # Deactivate the walker
        self.active = False
        return path

    def rotatePath(self, path, lx, ly, mode):
        # Rotate path coordinates by 180 degrees with respect to the
        # center of image with dimensions lx,ly
        pflipped = []
        for p in path:
            x, y = p
            xnew = lx - x
            if mode == 'rot':
                ynew = ly - y
            else:
                ynew = y

            pflipped.append([xnew, ynew])

        return pflipped

    def scanBottom(self, matrix, mode):
        # Scan from the bottom of the image to find a band which has
        # no boundary on the top
        print('Scanning Bottom with', mode)
        # Reset position of walker
        self.position = [0, 0]
        if mode == 'rot':
            # Rotate image by 180 deg
            transformed = np.rot90(matrix, 2)
        elif mode == 'flip':
            transformed = np.flipud(matrix)
        else:
            print('Unknown transfomation mode', mode)
            print('Use "flip" or "rot"')
            return
        # Scan from top
        path = self.scan(transformed, hold='l')
        d0, d1 = np.shape(matrix)

        # Check if the first launch point of the transformed image coincides with
        # the endpoint of the original image
        launch = self.launchpoints[-1][1]
        if mode == 'rot':
            endRot = d1 - self.endpoints[-2][1] - 1
        else:
            endRot = self.endpoints[0][1]
        if launch == endRot:
            # If points overlap, all bands have already been detected
            print('Overlapped - All bands detected')
            self.chirality = 'left'
            return
        else:
            # If points do not overlap, an additional band is present
            print('Lower End Detected')
            print(self.terminations[-1])
            self.terminations[-1] = self.rotatePath([self.terminations[-1]], d0, d1, mode)[0]
            self.chirality = 'right'
            self.paths.append(self.rotatePath(path, d0, d1, mode))
            self.sides.append('r')
            self.position = copy.copy(self.launchpoints[-1])
            # Move to next empty point
            while transformed[self.position[0], self.position[1]] > 0:
                self.position[1] += 1
            self.position[1] -= 1
            path = self.scan(transformed, hold='r')
            self.terminations[-1] = self.rotatePath([self.terminations[-1]], d0, d1, mode)[0]
            self.paths.append(self.rotatePath(path, d0, d1, mode))
            self.sides.append('l')

    def scanMultiple(self, matrix, plot=False):
        t0 = time.time()
        # Scan all lines in an image
        print('Scanning for Lines')
        # Containers for paths and if it's an inner or outer edge
        self.paths = []
        self.sides = []
        side = 'l'
        # Scan until ride edge of image has been reached
        while self.position[1] < np.shape(matrix)[1] - 1:
            # Scan a single path
            path = self.scan(matrix, hold=side)
            if path is None:
                # End of image reached by walker
                break
            # Append paths and sides to containers
            self.paths.append(path)
            self.sides.append(side)
            # Move walker to the last launchpoint
            self.position = copy.copy(self.launchpoints[-1])
            # Move to next empty point
            while matrix[self.position[0], self.position[1]] > 0:
                self.position[1] += 1
            # Move back onto the band
            self.position[1] -= 1

            # Flip the side
            if side == 'l':
                side = 'r'
            else:
                side = 'l'

        self.launchpointsRegular = self.launchpoints[:]
        self.endpointsRegular = self.endpoints[:]

        # Check for terminated outer bands on other side
        self.scanBottom(matrix, 'rot')

        self.launchpoints = self.launchpointsRegular[:]
        self.endpoints = self.endpointsRegular[:]

        # Check for terminated innter bands on other side
        self.scanBottom(matrix, 'flip')

        print('Detected', len(self.paths), 'Lines')

        if len(self.paths) % 2 != 0:
            print('Error: Line count uneven')

        if len(self.terminations) == 0:
            print('No Band Ends in this Image')
        elif len(self.terminations) == 2 or len(self.terminations) == 4:
            print('Band End Detected')
            self.containEnd = True
        else:
            print('Band termination error:', len(self.terminations), 'terminations')

        if not self.containEnd:
            self.chirality = None

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for side, path in zip(self.sides, self.paths):
                if side == 'l':
                    color = 'red'
                else:
                    color = 'blue'
                ax.plot(*list(zip(*path)), color=color, lw=0.1)
            ax.set_ylim([0, 5000])
            fig.savefig('img/out/walker.png', dpi=300)
        print('Linewalker completed in', str(round(time.time() - t0, 2)), 's')

    def fastFeatures(self, smoothing=1e4, plot=False):
        # Interpolate paths with an univariate spline with a smoothing term
        # to remove small features either generated through noise or
        # local features on the band
        print('Interpolating Fast Features')
        # smoothing = 0
        self.rsplines = []
        self.phis = []
        for path in self.paths:
            phi = [float(c[0]) for c in path]
            r = [float(c[1]) for c in path]
            spline = scipy.interpolate.UnivariateSpline(phi, r, s=smoothing)
            self.phis.append(phi)
            rs = spline(phi)
            self.rsplines.append(rs)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for rs, phi in zip(self.rsplines, self.phis):
                ax.plot(phi, rs, lw=0.2, color='blue')
            fig.savefig('img/out/gradients.jpg', dpi=300)

        # return radius and angle for the interpolated paths
        return self.rsplines, self.phis
