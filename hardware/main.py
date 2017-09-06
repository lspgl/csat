from subprocess import Popen, call, PIPE
import time


def release(n):
    c_str = 'gphoto2 --set-config burstnumber=1 --capture-image'
    call(c_str, shell=True)


def main():
    c_str = 'gphoto2 --set-config burstnumber=2 --capture-image-and-download --filename cpt%n.jpg --force-overwrite'
    #command = c_str.split()
    call('gphoto2 -D', shell=True)
    call(c_str, shell=True)
    #p = Popen(command, stdout=PIPE)


if __name__ == '__main__':
    main()
