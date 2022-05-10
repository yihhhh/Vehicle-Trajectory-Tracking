# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # ---------------|Lateral Controller|-------------------------
        # Use the results of linearization to create a state-space model
        A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,2*Ca*(lr-lf)/(m*xdot)] \
                ,[0,0,0,1],[0,(2*Ca)*(lr-lf)/(Iz*xdot),(2*Ca)*(lf-lr)/Iz, \
                (-2*Ca)*(lf**2 + lr**2)/(Iz*xdot)]])
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
        C = np.eye(4)
        D = np.zeros((4,1))

        # Discretize the system and extract Ad and Bd
        sysc = signal.StateSpace(A, B, C, D)
        sysd = sysc.to_discrete(delT)
        Ad = sysd.A
        Bd = sysd.B

        # Come up with reasonable values for Q and R (state and control weights)
        Q = np.array([[1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.01]]) # Weight important states more (e1)
        R = 50 # Don't want to oversteer on tight turns, plus smaller R is less stable

        # Find the closest node to the vehicle
        _, node = closestNode(X, Y, trajectory)

        # Choose a node that is ahead of our current node based on index
        forwardIndex = 150

        # Determine desired heading angle and e1 using two nodes - one ahead, and one closest
        # We use a try-except so we don't attempt to grab an index that is out of scope
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1], \
                                    trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 =  (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - \
                (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)

        except:
            # The index -1 represents the last element in that array (the end of the course)
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], \
                                    trajectory[-1,0]-trajectory[node,0])
            e1 =  (Y - trajectory[-1,1])*np.cos(psiDesired) - \
                (X - trajectory[-1,0])*np.sin(psiDesired)

        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot

        # Assemble error-based states into array
        states = np.array([e1,e1dot,e2,e2dot])

        # Solve discrete ARE and compute gain matrix
        S = np.matrix(linalg.solve_discrete_are(Ad, Bd, Q, R))
        K = np.matrix(linalg.inv(Bd.T @ S @ Bd + R) @ (Bd.T @ S @ Ad))
        #[l,v] = linalg.eig(Ad - Bd @ K)

        # Calculate delta via u = -Kx
        delta = wrapToPi((-K @ states)[0, 0])

        # ---------------|Longitudinal Controller|-------------------------
        # PID gains
        kp = 200
        ki = 10
        kd = 30

        # Reference value for PID to tune to
        desiredVelocity = 12

        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
