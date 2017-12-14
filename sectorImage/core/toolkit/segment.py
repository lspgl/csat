import numpy as np


class Segment:

    def __init__(self, phis, rs, imgNum, bandNum, identity=None):
        self.phis = phis
        self.rs = rs
        self.xs = self.rs * np.cos(self.phis)
        self.ys = self.rs * np.sin(self.phis)
        self.imgNum = imgNum
        self.bandNum = bandNum
        self.identity = identity

        self.lp = (phis[0], rs[0])
        self.ep = (phis[-1], rs[-1])

        self.comP = np.mean(phis)
        self.comR = np.mean(rs)
        self.com = (self.comP, self.comR)
