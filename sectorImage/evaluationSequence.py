import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from core.stitcher import Stitcher
import time


def EvaluationSequence(n, directory):
    t0 = time.time()
    fns = [directory + '/cpt' + str(i) + '.jpg' for i in range(1, n + 1)]
    s = Stitcher(fns, mpflag=True)
    s.loadImages()
    s.stitchImages()
    print('Completed in', round(time.time() - t0, 2), 's')
    return s
