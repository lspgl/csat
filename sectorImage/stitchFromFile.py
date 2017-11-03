import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from core.calibration import Calibrator
from core.stitcher import Stitcher
from core.toolkit.colors import Colors as _C
import time


def StitchFromFile(n=16, directory='hardware/combined'):
    t0 = time.time()
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    s = Stitcher(fns, calibration=None, mpflag=True)
    segments = s.loadSegments()
    s.combineSegments(segments)
    print(_C.BOLD + _C.CYAN + 'Computation Completed in ' + str(round(time.time() - t0, 2)) + 's' + _C.ENDC)
    return s

if __name__ == '__main__':
    StitchFromFile()
