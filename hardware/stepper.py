import time
import subprocess
from subprocess import Popen, call
import multiprocessing


HOST = "localhost"
PORT = 4223
STP = "6rHF7Q"
QRL = "ueV"

from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_stepper import BrickStepper

import random


class Stepper:

    def __init__(self):
        self.HOST = "localhost"
        self.PORT = 4223
        self.STP = "6rHF7Q"

        self.ipcon = IPConnection()
        self.stepper = BrickStepper(self.STP, self.ipcon)

        self.mode = 8  # Stepper mode, default is eights
        self.current = 1000  # in mA
        self.vmax = 10000  # Maximum velocity in steps/s
        self.ramping = 50000
        self.normTurn = 200  # Normalized steps per turn

    def enable(self):
        # Former tinkerconn
        # Connect to tinkerforge brick
        self.ipcon.connect(self.HOST, self.PORT)

        # Configure motor
        self.stepper.set_motor_current(self.current)
        self.stepper.set_step_mode(self.mode)
        self.stepper.set_max_velocity(self.vmax)
        self.stepper.set_speed_ramping(self.ramping, self.ramping)

        # Enable Coils to lock motor position
        self.stepper.enable()

        # Establish proper coil alignment
        self.stepper.set_steps(32)
        self.stepper.set_steps(-32)

        return

    def tinkerdisco(self):
        # Disable stepper and disconnect from tinkerforge brick
        self.stepper.disable()
        self.ipcon.disconnect()
        return

    def cb_position_reached(self, position):
        print('Reached Target')

    def tinkerstepper(self, n, synchfps=4):
        # Enable Laser only after a position is reached
        # Desynchronization can be easily seen if pictures have no laser line
        self.stepper.register_callback(self.stepper.CALLBACK_POSITION_REACHED, lambda x: self.cb_position_reached(x))

        time.sleep(0.45)

        # Calculate steps for n images
        steps = self.mode * self.normTurn / float(n)
        for i in range(0, n):
            # Rotate section
            self.stepper.set_steps(steps)

            # Wait FPS Time of camera
            time.sleep(1.0 / synchfps)

        return

if __name__ == '__main__':
    stepper = Stepper()
