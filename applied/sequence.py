import sys
import os
from functools import wraps
import time

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__ + '/../')

from tools.colors import Colors as _C

from hardware.camera import Camera
from hardware.stepper import Stepper
from hardware.coupledCapture import CoupledCapture

from electrode import Electrode
from pair import Pair
from environment import Environment as env

from sectorImage.calibrationSequence import CalibrationSequence
from sectorImage.evaluationSequence import EvaluationSequence
from sectorImage.combinedSequence import CombinedSequence

import h5py
import numpy as np
import datetime
import copy

import traceback


class Sequence:

    def __init__(self, offsite=False):
        """
        Measurement sequence class

        Parameters
        ----------
        offsite: bool, optional
            Determines if the system is operating with a connected setup or simulating the process from previously captured images
        """
        self.offsite = offsite
        print(_C.CYAN + _C.BOLD + 'Initializing sequence' + _C.ENDC)
        if self.offsite:
            self.prime = self._placeholder
            self.disable = self._placeholder
            self.calibrate = self.calibrateOffsite
            self.evaluate = self.evaluateOffsite
            self.measure = self.measureOffsite

        else:
            self.cam = Camera()
            if not self.cam.camera_available:
                print(_C.RED +
                      'No camera connected. Check USB and power connection.' +
                      _C.ENDC)
                sys.exit()

            self.stp = Stepper(autoEnable=False)
            print(_C.LIME + 'Ready for priming' + _C.ENDC)
        self.primed = False

    def _requiresPrimed(func):
        """
        Wrapper to ensure that the system is primed

        Parameters
        ----------
        func: function
            function to be wrapped

        Returns
        -------
        function or None
            wrapped func is returned if the system is primed, otherwise None
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.primed:
                return func(self, *args, **kwargs)
            else:
                print(_C.RED + _C.BOLD + 'System not primed' + _C.ENDC)
                print(_C.RED + 'Required for: ' + func.__name__ + _C.ENDC)
                return None

        return wrapper

    def _placeholder(self):
        return True

    def prime(self):
        """_requiresPrimed
        Priming the hardware
        Camera is triggered to ensure initial timing
        Stepper voltage is enabled
        """
        print(_C.CYAN + _C.BOLD + 'Priming camera and stepper' + _C.ENDC)
        # self.cam.fireSeries(n=3)
        self.stp.enable()
        time.sleep(0.1)
        if self.stp.enabled:
            print(_C.LIME + 'System ready' + _C.ENDC)
            self.primed = True
        else:
            print(_C.RED + 'Stepper failed to enable. Check USB connection.' +
                  _C.ENDC)
            sys.exit()

    def disable(self):
        """
        Turning of the stepper voltage
        """
        print(_C.CYAN + _C.BOLD + 'Disabling stepper' + _C.ENDC)
        self.stp.disable()
        self.calibrated = False
        print(_C.LIME + 'Stepper voltage disabled' + _C.ENDC)

    @_requiresPrimed
    def calibrate(self, n=16):
        print(_C.CYAN + _C.BOLD + 'Calibrating system' + _C.ENDC)
        input(
            _C.YEL +
            'Insert calibration piece and close door [Press any key when ready]' +
            _C.ENDC)
        CoupledCapture(n=n, directory='calib', stp=self.stp, cam=self.cam)
        oscillation = CalibrationSequence(n=n, directory='hardware/calib')
        self.calibrated = True
        return oscillation

    def calibrateOffsite(self, n=16, directory='hardware/combined'):
        print(_C.CYAN + _C.BOLD + 'Calibrating system from stored images' + _C.ENDC)
        oscillation = CalibrationSequence(n=n, directory=directory)
        self.calibrated = True
        return oscillation

    @_requiresPrimed
    def evaluate(self, n=16):
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        input(_C.YEL +
              'Insert Electrode and close door [Press any key when ready]' +
              _C.ENDC)
        CoupledCapture(n=n, directory='capture', stp=self.stp, cam=self.cam)
        stitcher = EvaluationSequence(n=n, directory='hardware/capture')
        return stitcher

    def evaluateOffsite(self, n=16, directory='hardware/combined'):
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        stitcher = EvaluationSequence(n=n, directory=directory)
        return stitcher

    def robustnessTest(self, n=16, n_iter=50):
        electrodes = []
        serial = 'RND-ROBUSTNESS'
        for i in range(n_iter):
            try:
                print('Iteration:', i + 1)
                CoupledCapture(n=n, directory='combined', stp=self.stp, cam=self.cam)
                spiral, calibration = CombinedSequence(n=n, directory='hardware/combined', env=env)
                localElectrode = Electrode(serial, spiral, calibration)
                electrodes.append(copy.copy(localElectrode))
            except KeyboardInterrupt:
                sys.exit()
            except:
                traceback.print_exc()
                # sys.exit()

        t = datetime.datetime.now()
        timestamp = (str(t.year) + '-' + str(t.month) + '-' + str(t.day) + '-' +
                     str(t.hour) + '-' + str(t.minute) + '-' + str(t.second))
        attributes = {'serial': serial,
                      'timestamp': timestamp,
                      'calib size': env.calib_size_mm,
                      'calib width': env.calib_width_mm,
                      }

        fn = 'CSAT_ROBUSTNESS.h5'
        with h5py.File(__location__ + '/data/' + fn, 'w') as f:
            for key in attributes:
                f.attrs[key] = attributes[key]
            for i, e in enumerate(electrodes):
                gname = 'ITER_' + str(i + 1)
                g = f.create_group(gname)
                g.create_dataset('spiral', data=[e.phis, e.rs])
                g.attrs['Scale'] = e.scale
                g.create_dataset('calibration', data=e.calibration)

        return electrodes

    @_requiresPrimed
    def measure(self, n=16, corrections=False):

        serial = input(_C.YEL +
                       'Insert 1st electrode with calibration ring and scan serial: ' +
                       _C.ENDC)
        electrodes = []
        times = []
        for i in range(2):
            if i == 1:
                input(_C.YEL + 'Insert 2nd electrode with calibration and press [Enter]')
            t0 = time.time()
            CoupledCapture(n=n, directory='combined', stp=self.stp, cam=self.cam)
            print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
            spiral, calibration = CombinedSequence(n=n, directory='hardware/combined', env=env)
            print(_C.CYAN + _C.BOLD + 'Measurement completed in ' + str(round(time.time() - t0, 2)) + 's' + _C.ENDC)

            localElectrode = Electrode(serial, spiral, calibration)
            electrodes.append(copy.copy(localElectrode))
            times.append(time.time() - t0)

        print(_C.CYAN + _C.BOLD + 'Pair completed in ' +
              str(round(sum(times), 2)) + 's' + _C.ENDC)

        pair = Pair(env=env, electrodes=tuple(electrodes), serial=serial, corrections=corrections)
        return pair

    def measureOffsite(self, n=16, directory1='hardware/positive', directory2='hardware/negative', corrections=False):
        serial = 'RD-OFFSITE'
        directories = [directory1, directory2]
        electrodes = []
        for i in range(2):
            print(_C.CYAN + _C.BOLD + 'Evaluating electrode ' + str(i + 1) + _C.ENDC)
            spiral, calibration = CombinedSequence(n=n, directory=directories[i], env=env)
            localElectrode = Electrode(serial, spiral, calibration)
            electrodes.append(copy.copy(localElectrode))

        # sys.exit()
        # print(_C.CYAN + _C.BOLD + 'Evaluating electrode 2' + _C.ENDC)
        # spiral, calibration = CombinedSequence(n=n, directory=directory2, env=env)
        # electrode2 = Electrode(serial, spiral, calibration)

        pair = Pair(env=env, electrodes=tuple(electrodes), serial=serial, corrections=corrections)
        return pair

    def calib_iter(self, n, n_iter):
        CoupledCapture(n=n, directory='combined', stp=self.stp, cam=self.cam)
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        CalibrationSequence(n=n, directory='hardware/combined', iteration=n_iter)

    def shuffle(self):
        import random
        nsteps = random.uniform(100, 1600)
        self.stp.stepper.enable()
        self.stp.stepper.set_steps(int(nsteps))
        time.sleep(2)
        self.stp.stepper.disable()

    def storeSpiral(self, spiral, fn=None):
        phis, rs, scale = spiral
        t = datetime.datetime.now()
        timestamp = (str(t.year) + '-' + str(t.month) + '-' + str(t.day) + '-' +
                     str(t.hour) + '-' + str(t.minute) + '-' + str(t.second))

        attributes = {'serial': 0,
                      'timestamp': timestamp,
                      'scale': scale,
                      }

        fn = 'CSAT_' + str(attributes['serial']) + '.h5'

        with h5py.File(__location__ + '/data/' + fn, 'w') as f:
            for key in attributes:
                f.attrs[key] = attributes[key]
            h5spiral = f.create_dataset('spiral', data=[phis, rs])
