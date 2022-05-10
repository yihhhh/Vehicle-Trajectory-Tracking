from controller import GPS, Gyro, Compass
import numpy as np
from util import clamp, wrapToPi

class BaseController():

    def __init__(self, trajectory):

        # Initialize variables
        self.trajectory = trajectory

        self.previousX = 0
        self.previousY = 0
        self.previousZ = 0
        self.previousPsi = 0

        self.previousXdotError = 0
        self.integralXdotError = 0


    def startSensors(self, timestep):

        # Instantiate objects and start up GPS, Gyro, and Compass sensors
        # For more details, refer to the Webots documentation
        self.gps = GPS("gps")
        self.gps.enable(timestep)

        self.gyro = Gyro("gyro")
        self.gyro.enable(timestep)

        self.compass = Compass("compass")
        self.compass.enable(timestep)

    def getStates(self, timestep):

        # Timestep returned by Webots is in ms, so we convert
        delT = 0.001*timestep

        # Extract (X, Y) coordinate from GPS
        position = self.gps.getValues()
        X = position[0]
        Y = position[1]

        # Find the rate of change in each axis, and store the current value of (X, Y)
        # as previous (X, Y) which will be used in the next call
        Xdot = (X - self.previousX)/(delT + 1e-9)
        self.previousX = X
        Ydot = (Y - self.previousY)/(delT + 1e-9)
        self.previousY = Y
        XYdot = np.array([[Xdot],[Ydot]])

        # Get heading angle and angular velocity
        psi = wrapToPi(self.getBearingInRad())
        angularVelocity = self.gyro.getValues()
        psidot = angularVelocity[2]

        # Get the rotation matrix (2x2) to convert velocities to the vehicle frame
        rotation_mat = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        xdot = (np.linalg.inv(rotation_mat) @ XYdot)[0,0]
        ydot = (np.linalg.inv(rotation_mat) @ XYdot)[1,0]

        # Clamp xdot above 0 so we don't have singular matrices
        xdot = clamp(xdot, 1e-5, np.inf)

        return delT, X, Y, xdot, ydot, psi, psidot
    
    def getBearingInRad(self):
        # Get compass relative north vector
        north = self.compass.getValues()

        # Calculate vehicle's heading angle from north
        rad = np.arctan2(north[1], north[0])

        # Convert to vehicle's heading angle from x-axis
        bearing = np.pi/2.0 - rad
        return bearing
