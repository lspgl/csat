import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence
from pair import Pair


def main():
    n_iters = 20
    s = Sequence(offsite=False)
    s.prime()
    for n_iter in range(n_iters):
        s.calib_iter(n=16, n_iters)
        s.shuffle()
    s.disable()


if __name__ == '__main__':
    main()
