import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence
from pair import Pair
import time


def main():
    n_iters = 20
    s = Sequence(offsite=False)
    s.prime()
    time.sleep(1)
    for n_iter in range(n_iters):
        s.calib_iter(16, n_iter)
        s.shuffle()
    s.disable()


if __name__ == '__main__':
    main()
