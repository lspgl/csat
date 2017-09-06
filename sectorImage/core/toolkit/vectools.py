# Vector calculation tools

import math


def connvector(p1, p2, norm=True):
    # Vector from p1 to p2

    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]

    if norm:
        n = math.sqrt(vx**2 + vy**2)
        if n == 0:
            return (0, 0)
        vx = vx / n
        vy = vy / n

    vector = (vx, vy)
    return vector


def pointdist(p1, p2):
    # Magnitude of distance between p1 and p2

    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def vectormags(v_arr):
    # Magnitudes of vectors in an array

    mags = []
    for vect in v_arr:
        mag_i = pointdist((0, 0), vect)
        mags.append(mag_i)
    return mags


def scalarproduct(v1, v2):
    # Dot product v1.v2
    v1x, v1y = v1
    v2x, v2y = v2

    scalar = (v1x * v2x) + (v1y * v2y)

    return scalar


def minimumconnector(p1, p2, pref):
    # Minimal distance between pref and line segment p1-p2
    x0, y0 = pref
    x1, y1 = p1
    x2, y2 = p2

    if p1 == p2:
        return pointdist(p1, pref)
    l2 = pointdist(p1, p2)**2
    pv = (x0 - x1, y0 - y1)
    wv = (x2 - x1, y2 - y1)
    t = max(0, min(1, scalarproduct(pv, wv) / l2))
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    projection = (projection_x, projection_y)

    dist = pointdist(pref, projection)

    return dist


if __name__ == '__main__':
    print('TESTING VECTOR TOOLS')

    testvect_a = (1, 1)
    testvect_b = (0, 1)

    comparator = [math.sqrt(2), 1.0]

    vectarr = [testvect_a, testvect_b]

    mags = vectormags(vectarr)
    assert mags == comparator
    print('vectormags OK')

    dist = pointdist(testvect_a, testvect_b)
    assert dist == 1
    print('pointdist OK')

    conn = connvector(testvect_a, testvect_b)
    assert conn == (-1, 0)
    print('connvector OK')

    s1 = scalarproduct(testvect_a, testvect_b)
    s2 = scalarproduct(testvect_b, testvect_a)
    assert s1 == s2
    assert s1 == 1
    print('scalarproduct OK')

    p1 = (0, 0)
    p2 = (1, 0)
    # Test along line segment
    assert minimumconnector(p1, p2, (0.5, 1)) == 1.0
    # Test outside line segment
    assert minimumconnector(p1, p2, (-1, 0)) == 1.0
    print('minimumconnector OK')

    print('VECTOR TOOLS OK')
