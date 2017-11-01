import sys
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

import stepper
import camera
from multiprocessing import Process
import time


def CoupledCapture(n, directory, stp, cam):
    kwargs = {'n': n, 'directory': __location__ + '/' + directory}
    #sp = Process(target=stp.discreteRotation, kwargs=kwargs)
    cp = Process(target=cam.collectSeries, kwargs=kwargs)
    # sp.start()
    cp.start()
    # delay = 0.5
    # time.sleep(delay)
    # stp.discreteRotation(**kwargs)
    timeshift = -0.05
    fps = 6
    t = n * (1 / fps) + timeshift
    stp.continuousRotation(t=t, nTurn=2)
    cp.join()
    print('done')


if __name__ == '__main__':
    cam = camera.Camera()
    stp = stepper.Stepper(autoEnable=True)
    CoupledCapture(16, 'capture', stp, cam)
    stp.disable()
