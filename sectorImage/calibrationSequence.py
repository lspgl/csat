import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from core.calibration import Calibrator
import time


def CalibrationSequence(n, directory):
    t0 = time.time()
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    c = Calibrator(fns)
    #c.computeMidpoint(fns[0], plot=True)
    c.computeAll()
    oscillation = c.oscillationCircle()
    c.plotCalibration()
    print('Calibration completed in', round(time.time() - t0, 2), 's')
    return oscillation


if __name__ == '__main__':
    CalibrationSequence(16, 'hardware/calib')
