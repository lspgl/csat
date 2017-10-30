import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from sequence import Sequence


def main():
    s = Sequence(offsite=True)
    s.prime()
    s.calibrate()
    s.evaluate()
    s.disable()


if __name__ == '__main__':
    main()
