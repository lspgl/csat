import numpy as np
import matplotlib.pyplot as plt
import time


def findMidpoints(binary):
    t0 = time.time()
    mask = np.roll(binary, shift=1)
    np.logical_xor(mask, binary, out=mask)
    rising = np.logical_and(binary, mask)
    np.logical_not(binary, out=binary)
    np.logical_and(binary, mask, out=binary)
    falling = np.roll(binary, shift=-1)
    logic = rising - falling

    print('Logical Rolling:', time.time() - t0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(logic)
    fig.savefig('img/out/logic.png', dpi=600)
