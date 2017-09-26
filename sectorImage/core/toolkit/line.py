import numpy as np
import math


class Line:

    def __init__(self, x1, y1, x2, y2, identifier=-1):
        self.identifier = identifier

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

        self.m = (y2 - y1) / (x2 - x1)
        self.m_vect = [x2 - x1, y2 - y1]
        self.q = y1 - self.m * x1

        # self.r, self.q, self.a, self.n = self.p2vect()

    def p2vect(self, confidence=1.0):
        # Calculate matrix and vector elements for point set
        # a = (x1,y1)T
        # n = ||1->2||T
        # r = I - nnT
        # q = r*a

        nx_raw = self.x2 - self.x1
        ny_raw = self.y2 - self.y1
        # Normalize
        length = math.sqrt(nx_raw**2 + ny_raw**2)
        nx = nx_raw / length
        ny = ny_raw / length

        scale = 3000
        n = [nx * scale, ny * scale]
        a = [self.x1, self.y1]
        nx2 = nx**2
        ny2 = ny**2
        nxy = nx * ny
        # Generate matrix
        r = confidence * np.array([[1 - nx2, -nxy], [-nxy, 1 - ny2]])
        # Generate vector
        q = confidence * np.array([[(1 - nx2) * self.x1 - nxy * self.y1], [-nxy * self.x1 + (1 - ny2) * self.y1]])
        return(r, q, a, n)
