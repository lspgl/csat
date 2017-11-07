from core import image, linewalker, singleImage, stitcher
from core.toolkit import pickler
import numpy as np
import time


class Environment:
    calib_size_mm = 68.0 / 2.0  # Outer radius of calibration piece
    calib_width_mm = 6.55   # Width of the calibration piece
    pitch_mm = 1.13  # Nominal electrode pitch


def main():

    t0 = time.time()

    #fns = ['img/src/cpt' + str(i) + '.jpg' for i in range(1, 23)]
    fns = ['../hardware/combined/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    # fns = ['../hardware/combined/cpt5.jpg']
    s = stitcher.Stitcher(fns, mpflag=False, env=Environment)
    s.loadImages()
    # s.pickleSave()

    #p = pickler.Pickler()
    #fn = 'stitcher.pkl'
    #s = p.load(fn)
    # s.stitchImages()

    print('Completed in', round(time.time() - t0, 2), 's')

    return


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
