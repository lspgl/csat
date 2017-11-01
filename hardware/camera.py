from subprocess import call, Popen, PIPE
import os
import readline
import signal
import time

FNULL = open(os.devnull, 'w')


class Camera:

    def __init__(self):
        self.checkConnection()

    def checkConnection(self):
        cmd = 'gphoto2 --auto-detect | grep usb'
        c = Popen(cmd, shell=True, stdout=PIPE)
        out = c.stdout.read()
        out = out.rstrip()
        if out == b'':
            print('No camera connected')
            self.camera_available = False
            return False
        else:
            print('Camera connected')
            self.camera_available = True
            return True

    def collectSingle(self, fn='single.jpg'):
        cmd = 'gphoto2 --set-config burstnumber=1 --force-overwrite --filename ' + fn + ' --capture-image-and-download'
        c = Popen(cmd, shell=True, stdout=PIPE)
        c.wait()

    def fireSeries(self, n=2):
        cmd = ['gphoto2',
               '--set-config', 'burstnumber=' + str(n),
               '--capture-image',
               '--filename=capture/cpt%n.jpg',
               '--capture-tethered']
        c = Popen(cmd, stdout=PIPE, stderr=PIPE)
        time.sleep(1)
        c.kill()
        cmd = ['gphoto2',
               '--delete-all-files']
        c = Popen(cmd, stdout=PIPE, stderr=PIPE)
        time.sleep(1)
        return

    def collectSeries(self, n=8, directory='capture'):

        cmd = ['gphoto2',
               '--set-config', 'burstnumber=' + str(n),
               '--capture-image-and-download',
               '--filename=' + directory + '/cpt%n.jpg',
               '--capture-tethered',
               '--force-overwrite']
        c = Popen(cmd, stdout=PIPE, stderr=PIPE)

        ctr = 0
        while True:
            # print('waiting for line...')
            line = c.stderr.readline()
            # print('recieved line')
            if line != '':
                ctr += 1
                print('Captured image', ctr)
            if ctr == n:
                c.kill()
                break

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
    cam.collectSeries(n=8)
