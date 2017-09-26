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

    def __init__(self, autoEnable=False):
        self.HOST = "localhost"
        self.PORT = 4223
        self.STP = "6rHF7Q"

        self.ipcon = IPConnection()
        self.stepper = BrickStepper(self.STP, self.ipcon)

        self.mode = 8  # Stepper mode, default is eights
        self.current = 1000  # in mA
        self.vmax = 10000  # Maximum velocity in steps/s
        self.ramping = 30000
        self.normTurn = 200  # Normalized steps per turn
        if autoEnable:
            self.enable()
        else:
            self.enabled = False

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
        time.sleep(0.1)
        self.stepper.set_steps(-32)

        self.enabled = True

        return

    def disable(self):
        # Disable stepper and disconnect from tinkerforge brick
        self.stepper.disable()
        self.ipcon.disconnect()
        self.enabled = False
        return

    def cb_position_reached(self, position):
        print('Reached Target')

    def discreteRotation(self, n):
        # Enable Laser only after a position is reached
        # Desynchronization can be easily seen if pictures have no laser line
        self.stepper.register_callback(self.stepper.CALLBACK_POSITION_REACHED, lambda x: self.cb_position_reached(x))

        # time.sleep(0.55)
        time.sleep(0.18)

        # Calculate steps for n images
        steps = int(self.mode * self.normTurn // float(n))
        print(steps)
        for i in range(0, n):
            # Rotate section
            self.stepper.set_steps(steps)
            # Wait FPS Time of camera and release time (calibrated for shutter of 1/200)
            deadtime = 1.0 / 5.88
            deadtime = 1.0/ 6.0
            time.sleep(deadtime)
        return

    def continuousRotation(self, t):
        totalSteps = self.normTurn * self.mode
        speed = int(totalSteps/t)
        self.stepper.set_max_velocity(speed)
        self.stepper.set_steps(totalSteps)
        time.sleep(t)
        self.stepper.set_max_velocity(self.vmax)
        return



if __name__ == '__main__':
    stepper = Stepper()
    stepper.enable()
    stepper.discreteRotation(8)
    stepper.disable()
