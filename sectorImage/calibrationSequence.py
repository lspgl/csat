from core.calibration import Calibrator
import time


def CalibrationSequence(n, directory):
    t0 = time.time()
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    c = Calibrator(fns)
    c.computeAll()
    oscillation = c.oscillationCircle()
    print('Calibration completed in', round(time.time() - t0, 2), 's')
    return oscillation
