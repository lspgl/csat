from core import image, linewalker, singleImage, stitcher
from core.toolkit import pickler
import numpy as np
import time


def main():

    t0 = time.time()

    #fns = ['img/src/cpt' + str(i) + '.jpg' for i in range(1, 23)]
    fns = ['../hardware/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    fns = ['../hardware/cpt16.jpg']
    s = stitcher.Stitcher(fns, mpflag=False)
    s.loadImages()
    # s.pickleSave()

    #p = pickler.Pickler()
    #fn = 'stitcher.pkl'
    #s = p.load(fn)
    s.stitchImages()

    print('Completed in', round(time.time() - t0, 2), 's')

    return


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
