import sys
import os
from functools import wraps

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__ + '/../')

from tools.colors import Colors as _C

from hardware.camera import Camera
from hardware.stepper import Stepper
from hardware.coupledCapture import CoupledCapture

from sectorImage.calibrationSequence import CalibrationSequence
from sectorImage.evaluationSequence import EvaluationSequence
from sectorImage.combinedSequence import CombinedSequence


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
        self.calibrated = False

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

    def _requiresCalibrated(func):
        """
        Wrapper to ensure that the system is calibrated

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
            if self.calibrated:
                return func(self, *args, **kwargs)
            else:
                print(_C.RED + _C.BOLD + 'System not calibrated' + _C.ENDC)
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
    @_requiresCalibrated
    def evaluate(self, n=16):
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        input(_C.YEL +
              'Insert Electrode and close door [Press any key when ready]' +
              _C.ENDC)
        CoupledCapture(n=n, directory='capture', stp=self.stp, cam=self.cam)
        stitcher = EvaluationSequence(n=n, directory='hardware/capture')
        return stitcher

    @_requiresCalibrated
    def evaluateOffsite(self, n=16, directory='hardware/combined'):
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        stitcher = EvaluationSequence(n=n, directory=directory)
        return stitcher

    @_requiresPrimed
    def measure(self, n=16):
        input(_C.YEL +
              'Insert electrode with calibration ring and close door [Press any key when ready]' +
              _C.ENDC)
        CoupledCapture(n=n, directory='combined', stp=self.stp, cam=self.cam)
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        stitcher = CombinedSequence(n=n, directory='hardware/combined')
        return stitcher

    def measureOffsite(self, n=16, directory='hardware/combined'):
        print(_C.CYAN + _C.BOLD + 'Evaluating electrode' + _C.ENDC)
        stitcher = CombinedSequence(n=n, directory=directory)
        return stitcher
