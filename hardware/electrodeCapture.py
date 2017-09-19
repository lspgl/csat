import stepper
import camera
from multiprocessing import Process
import time


def captureTurn(n, stp, cam):
    kwargs = {'n': n}
    #sp = Process(target=stp.discreteRotation, kwargs=kwargs)
    cp = Process(target=cam.collectSeries, kwargs=kwargs)
    # sp.start()
    cp.start()
    delay = 0.5
    time.sleep(delay)
    stp.discreteRotation(**kwargs)


if __name__ == '__main__':
    cam = camera.Camera()
    stp = stepper.Stepper(autoEnable=True)
    captureTurn(16, stp, cam)
    stp.disable()
