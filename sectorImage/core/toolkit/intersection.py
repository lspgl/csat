
def Intersection(ln1, ln2):
    x1 = ln1.x1
    y1 = ln1.y1
    x2 = ln1.x2
    y2 = ln1.y2

    x3 = ln2.x1
    y3 = ln2.y1
    x4 = ln2.x2
    y4 = ln2.y2

    if (max(x1, x2) < min(x3, x4)):
        return False
    A1 = (y1 - y2) / (x1 - x2)
    A2 = (y3 - y4) / (x3 - x4)
    b1 = y1 - A1 * x1
    b2 = y3 - A2 * x3

    if (A1 == A2):
        return False
    Xa = (b2 - b1) / (A1 - A2)
    if Xa < max(min(x1, x2), min(x3, x4)) or Xa > min(max(x1, x2), max(x3, x4)):
        return False
    else:
        return True
