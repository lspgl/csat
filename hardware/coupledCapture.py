import stepper
import camera
from multiprocessing import Process
import time


def CoupledCapture(n, directory, stp, cam):
    kwargs = {'n': n, 'directory': directory}
    #sp = Process(target=stp.discreteRotation, kwargs=kwargs)
    cp = Process(target=cam.collectSeries, kwargs=kwargs)
    # sp.start()
    cp.start()
    delay = 0.5
    time.sleep(delay)
    # stp.discreteRotation(**kwargs)
    timeshift = -0.05
    fps = 6
    t = n * (1 / fps) + timeshift
    stp.continuousRotation(t=t)


if __name__ == '__main__':
    cam = camera.Camera()
    stp = stepper.Stepper(autoEnable=True)
    CoupledCapture(16, 'capture', stp, cam)
    stp.disable()
