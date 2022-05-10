import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

class CustomController(BaseController):

    def __init__(self, trajectory, desired_velocity):

        super().__init__(trajectory)

        # System constants
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        # Lateral controller parameters
        self.forward_index = 150
        self.Q = np.array([[1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.01]]) 
        self.R = 50 
        
        # Longitudinal controller parameters
        self.desired_velocity = 12
        self.Kp = 200
        self.Ki = 10
        self.Kd = 30

    def update(self, timestep):

        trajectory = self.trajectory
        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        forwardIndex = self.forward_index
        
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        # Lateral LQR Controller        
        Q = self.Q
        R = self.R
        
        A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,2*Ca*(lr-lf)/(m*xdot)] \
                ,[0,0,0,1],[0,(2*Ca)*(lr-lf)/(Iz*xdot),(2*Ca)*(lf-lr)/Iz, \
                (-2*Ca)*(lf**2 + lr**2)/(Iz*xdot)]])
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
        C = np.eye(4)
        D = np.zeros((4,1))

        sysc = signal.StateSpace(A, B, C, D)
        sysd = sysc.to_discrete(delT)
        Ad = sysd.A
        Bd = sysd.B

        _, node = closestNode(X, Y, trajectory)
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1], \
                                    trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 =  (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - \
                (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)
        except:
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], \
                                    trajectory[-1,0]-trajectory[node,0])
            e1 =  (Y - trajectory[-1,1])*np.cos(psiDesired) - \
                (X - trajectory[-1,0])*np.sin(psiDesired)

        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot

        error_states = np.array([e1,e1dot,e2,e2dot])

        S = np.matrix(linalg.solve_discrete_are(Ad, Bd, Q, R))
        K = np.matrix(linalg.inv(Bd.T @ S @ Bd + R) @ (Bd.T @ S @ Ad))

        delta = wrapToPi((-K @ error_states)[0, 0])

        # Longitudinal PID Controller
        kp = self.Kp
        ki = self.Ki
        kd = self.Kd
        desiredVelocity = self.desired_velocity

        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        return X, Y, xdot, ydot, psi, psidot, F, delta
