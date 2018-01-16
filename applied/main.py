import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence
from pair import Pair

import traceback


def main():
    s = Sequence(offsite=True)
    s.prime()
    continuous = False
    if continuous:
        while True:
            try:
                pair = s.measure(n=16)
                pair.store()
                del pair
            except KeyboardInterrupt:
                s.disable()
                sys.exit()
            except:
                del pair
                s.disable()
                traceback.print_exc()
    else:
        pair = s.measure(n=16)
        pair.store()
    # for i in range(16):
    #    pair = Pair('CSAT_' + str(i + 1) + '.h5', fromFile=True, corrections=True)
    #    pair.plot()
    s.disable()


if __name__ == '__main__':
    main()
