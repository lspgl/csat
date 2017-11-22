import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from core.calibration import Calibrator
from core.stitcher import Stitcher
from core.sources import Sources
from core.toolkit.colors import Colors as _C
import time


def CalibrationSequence(n, directory, iteration=None):
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    Sources(fns)
    c = Calibrator(fns)
    c.computeAll()
    c.oscillationCircle()
    c.plotCalibration(fn=iteration)
    return


if __name__ == '__main__':
    CalibrationSequence(16, 'hardware/calib')
