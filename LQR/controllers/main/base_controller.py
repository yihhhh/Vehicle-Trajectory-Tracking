from controller import Robot, GPS, Gyro, Compass, Receiver
import numpy as np
import struct
from util import clamp, wrapToPi

class BaseController():

    def __init__(self, trajectory):

        self.trajectory = trajectory
        self.previousX = 0
        self.previousY = 0
        self.previousZ = 0
        self.previousPsi = 0

        self.previousXdotError = 0
        self.integralXdotError = 0

    def startSensors(self, timestep):
        self.gps = GPS("gps")
        self.gps.enable(timestep)

        self.gyro = Gyro("gyro")
        self.gyro.enable(timestep)

        self.compass = Compass("compass")
        self.compass.enable(timestep)

    def getStates(self, timestep):

        delT = 0.001*timestep

        position = self.gps.getValues()
        X = position[0]
        Y = position[1]

        Xdot = (X - self.previousX)/(delT + 1e-9)
        self.previousX = X
        Ydot = (Y - self.previousY)/(delT + 1e-9)
        self.previousY = Y
        XYdot = np.array([[Xdot],[Ydot]])

        psi = wrapToPi(self.getBearingInRad())
        angularVelocity = self.gyro.getValues()
        psidot = angularVelocity[2]

        rotation_mat = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        xdot = (np.linalg.inv(rotation_mat) @ XYdot)[0,0]
        ydot = (np.linalg.inv(rotation_mat) @ XYdot)[1,0]

        xdot = clamp(xdot, 1e-5, np.inf)

        return delT, X, Y, xdot, ydot, psi, psidot
    
    def getBearingInRad(self):
        north = self.compass.getValues()
        rad = np.arctan2(north[1], north[0])
        bearing = np.pi/2.0 - rad
        return bearing
