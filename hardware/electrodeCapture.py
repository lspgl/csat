import stepper
import camera
import subprocess


def captureTurn(n, stp, cam):
    cam.collectSeries(n=n, nowait=True)
    stp.discreteRotation(n=n)

if __name__ == '__main__':
    cam = camera.Camera()
    stp = stepper.Stepper(autoEnable=True)
    captureTurn(8, stp, cam)
    stp.disable()
