import cv2
import time
import numpy as np


class Sources:

    def __init__(self, fns):
        t0 = time.time()
        self.fns = fns
        self.srcs = [(fn, cv2.imread(fn, cv2.IMREAD_GRAYSCALE)) for fn in self.fns]
        for src in self.srcs:
            fn = src[0].split('.')[0] + '.npy'
            print(fn)
            np.save(fn, src[1])
        print('image array:', time.time() - t0)
