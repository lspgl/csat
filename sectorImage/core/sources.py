import cv2
import time
import numpy as np
from .toolkit.colors import Colors as _C


class Sources:

    def __init__(self, fns):
        t0 = time.time()
        self.fns = fns
        self.srcs = [(fn, cv2.imread(fn, cv2.IMREAD_GRAYSCALE)) for fn in self.fns]
        for src in self.srcs:
            fn = src[0].split('.')[0] + '.npy'
            np.save(fn, src[1])
        print(_C.CYAN + 'File preparation: ' + str(round(time.time() - t0, 2)) + _C.ENDC)
