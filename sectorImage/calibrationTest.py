from core import calibration
import time


def main():

    t0 = time.time()

    fns = ['img/src/cpt' + str(i) + '.jpg' for i in range(1, 9)]
    #fns = ['img/src/cpt1.jpg']
    c = calibration.Calibrator(fns)
    # c.sweepFFT()
    c.sweepAll()
    print('Completed in', round(time.time() - t0, 2), 's')

    return


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
