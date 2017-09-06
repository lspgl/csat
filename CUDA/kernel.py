import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2


def main():
    laplacian_pts = '''
    -4 -1 0 -1 -4
    -1 2 3 2 -1
    0 3 4 3 0
    -1 2 3 2 -1
    -4 -1 0 -1 -4
    '''.split()

    laplacian = np.array(laplacian_pts, dtype=np.float32).reshape(5, 5)
    response = np.zeros((50, 50))
    response[20:25, 20:25] = laplacian

    trafo = fft2(response)
    print(trafo)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(response)

    fig.savefig('response_2')


if __name__ == '__main__':
    main()
