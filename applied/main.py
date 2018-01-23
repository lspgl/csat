import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence
from pair import Pair

import traceback


def main():
    sysargs = []

    sysargs.extend(['-s', '--single'])
    if ('-s' in sys.argv) or ('--single' in sys.argv):
        continuous = False
    else:
        continuous = True

    sysargs.extend(['-c', '--corrections'])
    if ('-c' in sys.argv) or ('--corrections' in sys.argv):
        corrections = True
    else:
        corrections = False

    sysargs.extend(['--offsite'])
    if '--offsite' in sys.argv:
        offsite = True
        continuous = False
    else:
        offsite = False

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg not in sysargs:
                print('Valid sysargs:')
                [print(a) for a in sysargs]
                print()
                print('Supplied sysargs:')
                [print(a) for a in sys.argv[1:]]
                print()
                raise Exception('Invalid Sysarg')

    s = Sequence(offsite=offsite)
    s.prime()
    if continuous:
        while True:
            try:
                pair = s.measure(n=16, corrections=corrections)
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
        pair = s.measure(n=16, corrections=corrections)
        pair.store()

    s.disable()


if __name__ == '__main__':
    main()
