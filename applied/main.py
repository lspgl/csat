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
    # pair = s.measure(n=16)
    # pair.store()
    # pair.load()
    pair = Pair('CSAT_LabTest.h5', fromFile=True)
    pair.computeGap()
    pair.plot()
    # s.storeSpiral(spiral)
    s.disable()


if __name__ == '__main__':
    main()
