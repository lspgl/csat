from subprocess import call, Popen, PIPE
import os

FNULL = open(os.devnull, 'w')


class Camera:

    def __init__(self, a=0,):
        self.checkConnection()
        if self.camera_available:
            self.wipe()

    def checkConnection(self):
        cmd = 'gphoto2 --auto-detect | grep usb'
        c = Popen(cmd, shell=True, stdout=PIPE)
        out = c.stdout.read()
        out = out.rstrip()
        if out == b'':
            print('No camera connected')
            self.camera_available = False
        else:
            print('Camera connected')
            self.camera_available = True

    def wipe(self):
        cmd = 'gphoto2 -D'
        c = Popen(cmd, shell=True, stdout=PIPE).wait()

    def collectSingle(self, fn='single.jpg'):
        cmd = 'gphoto2 --capture-image-and-download --force-overwrite --filename ' + fn
        c = Popen(cmd, shell=True, stdout=PIPE)
        c.wait()

    def collectSeries(self, n=2):
        cmd = ('gphoto2 --set-config burstnumber=' +
               str(n) + ' --force-overwrite --filename cpt%n.jpg --capture-image-and-download')
        c = Popen(cmd, shell=True, stdout=PIPE)
        c.wait()

    def getBattery(self):
        cmd = ('gphoto2 --get-config batterylevel')
        c = Popen(cmd, shell=True, stdout=PIPE)
        out = c.stdout.read()
        self.batterylevel = float(out.decode("utf-8").splitlines()[-1].split(':')[-1].strip()[:-1]) / 100.0
        print('Current charge:', self.batterylevel * 100, '%')
        return self.batterylevel


if __name__ == '__main__':
    cam = Camera()
    cam.getBattery()
    # cam.collectSingle()
    # cam.collectSeries()
