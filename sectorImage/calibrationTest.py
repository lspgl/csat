from core.calibration import Calibrator
import time


def main():

    t0 = time.time()

    fns = ['img/src/calibration_new_light/cpt' + str(i) + '.jpg' for i in range(1, 17)]
    fns = ['img/src/calibration_new_light/cpt8.jpg']
    # fns = ['../hardware/cpt' + str(i) + '.jpg' for i in range(1, 17)]

    c = Calibrator(fns)
    c.computeMidpoint(fns[0], plot=True)
    # c.computeAll()
    c.loadCalibration('data/calibration.npy')
    c.plotCalibration()

    #mps = c.getMidpoints(mp_FLAG=False)
    # mps = c.loadMidpoints()
    # c.plotMidpoints(mps)
    print('Completed in', round(time.time() - t0, 2), 's')

    return


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
