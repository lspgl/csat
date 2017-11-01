import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from core.calibration import Calibrator
from core.stitcher import Stitcher
from core.toolkit.colors import Colors as _C
import time


def CombinedSequence(n, directory):
    t0 = time.time()
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    c = Calibrator(fns)
    calibration = c.computeAll()
    # oscillation = c.oscillationCircle()
    # print(oscillation)
    # c.plotCalibration()
    print(_C.BOLD + _C.CYAN + 'Calibration completed in ' + str(round(time.time() - t0, 2)) + 's' + _C.ENDC)
    t1 = time.time()
    s = Stitcher(fns, calibration=calibration, mpflag=True)
    s.loadImages()
    s.stitchImages(plot=True)
    print(_C.BOLD + _C.CYAN + 'Detection completed in ' + str(round(time.time() - t1, 2)) + 's' + _C.ENDC)
    print(_C.BOLD + _C.CYAN + 'Computation Completed in ' + str(round(time.time() - t0, 2)) + 's' + _C.ENDC)
    return s


if __name__ == '__main__':
    CombinedSequence(n=16, directory='hardware/combined')
