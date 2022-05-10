# Fill the respective function to implement the PID controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# Custom Controller Class
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)
        
        # Initialize necessary variables
        self.integralPsiError = 0
        self.previousPsiError = 0
        self.previousXdotError = 0

    def update(self, timestep):

        trajectory = self.trajectory

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)


        # ---------------|Lateral Controller|-------------------------
        # Find the closest node to the vehicle
        _, node = closestNode(X, Y, trajectory)

        # Choose a node that is ahead of our current node based on index
        forwardIndex = 50
        
        # Two distinct ways to calculate the desired heading angle:
        # 1. Find the angle between a node ahead and the car's current position
        # 2. Find the angle between two nodes - one ahead, and one closest
        # The first method has better overall performance, as the second method 
        # can read zero error when the car is not actually on the trajectory
        
        # We use a try-except so we don't attempt to grab an index that is out of scope
        # 1st method
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-Y, \
                                   trajectory[node+forwardIndex,0]-X)
        except:
            psiDesired = np.arctan2(trajectory[-1,1]-Y, \
                                  trajectory[-1,0]-X)
        
        # 2nd method                          
        # try:
            # psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1], \
                    # trajectory[node+forwardIndex,0]-trajectory[node,0])
        # except:
            # psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], \
                    # trajectory[-1,0]-trajectory[node,0])
        # PID gains
        kp = 1
        ki = 0.005
        kd = 0.001

        # Calculate difference between desired and actual heading angle
        psiError = wrapToPi(psiDesired-psi)

        self.integralPsiError += psiError
        derivativePsiError = psiError - self.previousPsiError
        delta = kp*psiError + ki*self.integralPsiError*delT + kd*derivativePsiError/delT
        delta = wrapToPi(delta)

        # ---------------|Longitudinal Controller|-------------------------
        # PID gains
        kp = 200
        ki = 10
        kd = 30

        # Reference value for PID to tune to
        desiredVelocity = 6

        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
