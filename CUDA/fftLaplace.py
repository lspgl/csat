import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from scipy import ndimage
from scipy.signal import fftconvolve
import time


def fftlaplace(image, mask=None):
    if mask is None:
        laplacian_pts = '''
        -4 -1 0 -1 -4
        -1 2 3 2 -1
        0 3 4 3 0
        -1 2 3 2 -1
        -4 -1 0 -1 -4
        '''.split()

        laplacian = np.array(laplacian_pts, dtype=np.float32).reshape(5, 5)
    else:
        laplacian = np.array(mask)
    kernel = np.zeros_like(image)
    kernel[:laplacian.shape[0], :laplacian.shape[1]] = laplacian
    print(kernel)

    im_fft = np.absolute(fft2(image))
    print('image fft done')
    kernel_fft = np.absolute(fft2(kernel))
    print('kernel fft done')

    process_fft = np.multiply(im_fft, kernel_fft)
    print('Fourier Space piecwise multiplication done')

    process = np.absolute(fft2(process_fft))
    print('Inverse Fourier Tranform')

    return process


def main():
    fn = 'cpt1.jpg'
    image = ndimage.imread(fn, flatten=True)
    t0 = time.time()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    print('process start')
    kernel = [[1, -2, 1]]

    #laplacian_naive = np.abs(ndimage.convolve1d(image, kernel, axis=1))
    laplacian_naive = np.abs(ndimage.convolve(image, kernel))
    print (laplacian_naive)
    ax1.imshow(laplacian_naive)
    t1 = time.time()
    print(t1 - t0)
    kernel = [[1, -2, 1]]
    #laplacian_fast = fftlaplace(image, mask=kernel)
    laplacian_fast = np.abs(fftconvolve(image, kernel))
    ax2.imshow(laplacian_fast)
    print(time.time() - t1)
    fig.savefig('comparison', dpi=300)
if __name__ == '__main__':
    main()
