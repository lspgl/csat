

class Electrode:

    def __init__(self, serial, spiral, calibration):
        self.phis, self.rs, self.scale, self.chirality = spiral
        self.calibration = calibration
        self.serial = serial
