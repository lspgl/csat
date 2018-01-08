import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence
from pair import Pair


def main():
    s = Sequence(offsite=True)
    s.prime()
    #pair = s.measure(n=16)
    # pair.store()
    pair = Pair('CSAT_RD-OFFSITE.h5', fromFile=True, corrections=True)
    pair.plot()
    s.disable()


if __name__ == '__main__':
    main()
