import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from os import listdir
from os.path import isfile, join


from pair import Pair


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '-m':
            for i in range(16):
                pair = Pair('CSAT_' + str(i + 1) + '.h5', fromFile=True, corrections=True)
                pair.plot()
        else:
            for arg in sys.argv[1:]:
                print('Correcting', arg)
                pair = Pair(arg, fromFile=True, corrections=True)
    else:
        dirfiles = [f for f in listdir(__location__ + '/data')
                    if isfile(join(__location__ + '/data', f)) and f[-3:] == '.h5']
        for fn in dirfiles:
            print('Correcting', fn)
            pair = Pair(fn, fromFile=True, corrections=True)


if __name__ == '__main__':
    main()
