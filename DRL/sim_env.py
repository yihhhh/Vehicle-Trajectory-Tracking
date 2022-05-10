import os, sys, time
import subprocess
import numpy as np
from scipy import signal
from gym import spaces

from controller import GPS, Gyro, Compass
from controller import Display
from vehicle import Driver

# Change as your own path
vehicle_proj_dir = 'C:/Users/yihan/myWorkspace/16745/16745-Project/DRL'

sys.path.insert(0, os.path.join(vehicle_proj_dir, 'controllers/main'))
from util import getTrajectory, wrapToPi, closestNode

# load webot world & trajectory
webots_world_path = os.path.join(vehicle_proj_dir, 'worlds/automotive_new.wbt')
traj = getTrajectory(os.path.join(vehicle_proj_dir, 'controllers/main/buggyTrace.csv'))


class TracingEnv:
    def __init__(self, trajectory=traj):

        self._set_driver()
        self.console = self.driver.getDisplay("console")
        self.console.setFont("Arial Black", 14, True)
        self.timestep = int(self.driver.getBasicTimeStep())
        # print(self.timestep)
        self._start_sensors(self.timestep)

        # init coordinate variables
        self.trajectory = trajectory

        self.previousX = 0
        self.previousY = 0
        self.previousZ = 0
        self.previousPsi = 0

        self.use_lm = False

        # init gym-env variables
        self.steps = 0
        self.passMiddlePoint = False
        self.lastNearIdx = 0

        self.desiredVelocity = 6
        
        self.XVec = []
        self.YVec = []

        # gym attributions
        self.action_space = spaces.Box(low = -np.ones(2), high = np.ones(2), dtype=np.float)
        self.observation_space = spaces.Box(low = -np.ones(5)* float('inf'), high = np.ones(5)* float('inf'), dtype=np.float)
        

    def start_webots(self):
        """
        start webots
        """
        command = ['webots', '--mode=fast', '--stdout', '--stderr', webots_world_path]
        self.p_webots = subprocess.Popen(command, preexec_fn=os.setsid)
        # wait for several seconds in case error occurred
        time.sleep(3)
        signal.signal(signal.SIGINT, self.manage_ctrlC)

    def manage_ctrlC(self, *args):
        print("***************** Closing Webots... *********************")
        self.close()
        print("******************* Successfully Exit *********************")

    def _set_driver(self):
        self.driver = Driver()
        self.driver.setDippedBeams(True)
        self.driver.setGear(1) # Torque control mode

    def _ConsoleUpdate(self, disError, psierr, vel):
        # refresh
        console = self.console
        x = console.getWidth()
        y = console.getHeight()
        console.setAlpha(1.0)
        console.setColor(0x000000)
        console.fillRectangle(0, 0, x, y)
        console.setColor(0xFFFFFF)

        console.drawText(
            "\n\nDistance err: " + str(round(disError,5)) + \
            "\n\nOrientation err: " + str(round(psierr,3)) + \
            "\n\nLongitude Vel: " + str(round(vel,3)), 
            5, 5
        )
    
    def _start_sensors(self, timestep):
        # Instantiate objects and start up GPS, Gyro, and Compass sensors
        # For more details, refer to the Webots documentation
        self.gps = GPS("gps")
        self.gps.enable(timestep)

        self.gyro = Gyro("gyro")
        self.gyro.enable(timestep)

        self.compass = Compass("compass")
        self.compass.enable(timestep)

    def getBearingInRad(self):
        # Get compass relative north vector
        north = self.compass.getValues()

        # Calculate vehicle's heading angle from north
        rad = np.arctan2(north[1], north[0])

        # Convert to vehicle's heading angle from x-axis
        bearing = np.pi/2.0 - rad
        return bearing

    # return X, Y, xdot, ydot, psi, psidot
    def _get_states(self, timestep):
        
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

        # clip xdot above 0 so we don't have singular matrices
        xdot = np.clip(xdot, 1e-5, np.inf)

        return delT, X, Y, xdot, ydot, psi, psidot

    def _get_obs(self):
        pass
    
    def _get_reward(self, longit_vel, psierr, distance):
        r = 0.0 
        # reward has 3 parts
        # - bouns for run along trajectory
        r += 0.1 * min(longit_vel, 10.0)
        # - penalty for orientation error
        r -= 0.2 * longit_vel * np.square(psierr)
        # - bouns/penalty for distance from central line
        if distance <= 5.0:
            r += 0.1 * (5.0 - distance)
        else:
            r += -0.4 * (distance - 5.0)
        return r

    # def _get_cost(self, disterr):
    #     if disterr < 5.0:
    #         return 0.0
    #     elif disterr < 10.0:
    #         return 0.2 * (disterr - 5.0)
    #     else:
    #         return 1.0

    def reset(self):
        self.steps = 0
        self.passMiddlePoint = False
        self.lastNearIdx = 0
        self.driver.simulationReset()

        # self.driver.step() 
        obs = self.step([0.0, 0.0])[0]

        return obs


    def step(self, action):
        
        self.steps += 1

        # actions: 2-dim
        # - a_0: control engine force; \in [-1.0, 1.0] (corresponding to [0, 0.3*max_force])
        # - a_1: control steering angle; \in [-1.0, 1.0] (corresponding to [-pi/4, pi/4])
        f = (action[0] + 1.0) / 2  * 0.3  # to bound it in [0.0, 0.3]
        delta = action[1] * np.radians(45) 
        self.driver.setThrottle(np.clip(f, 0, 1))
        self.driver.setSteeringAngle(-np.clip(delta, np.radians(-45), np.radians(45)))
        
        for _ in range(5):
            self.driver.step()
        
        delT, X, Y, xdot, ydot, psi, psidot = self._get_states(self.timestep * 5)
        disError, nearIdx = closestNode(X, Y, self.trajectory)

        stepToMiddle = nearIdx - len(self.trajectory)/2.0
        if not self.passMiddlePoint and abs(stepToMiddle) < 100.0:
            self.passMiddlePoint = True
    
        if not self.passMiddlePoint and (nearIdx > len(self.trajectory)/2.0):
            nearIdx = 0

        # 1. Compute observations
        # Observation: 5-dim 
        # - longit_vel : longitude velocity along trajectory
        # - e1      : distance from vehicle to nearest reference point 
        # - e1dot   : d(e_1)/dt 
        # - e2      : orientation error with reference trajectory
        # - e2dot   : d(e_2)/dt
        if nearIdx + 150 < len(self.trajectory):
            psiDesired = np.arctan2(self.trajectory[nearIdx+150,1]-self.trajectory[nearIdx,1], \
                                    self.trajectory[nearIdx+150,0]-self.trajectory[nearIdx,0])
            e1 =  (Y - self.trajectory[nearIdx+150,1])*np.cos(psiDesired) - \
                (X - self.trajectory[nearIdx+150,0])*np.sin(psiDesired)

        else:
            # The index -1 represents the last element in that array (the end of the course)
            psiDesired = np.arctan2(self.trajectory[-1,1]-self.trajectory[nearIdx,1], \
                                    self.trajectory[-1,0]-self.trajectory[nearIdx,0])
            e1 =  (Y - self.trajectory[-1,1])*np.cos(psiDesired) - \
                (X - self.trajectory[-1,0])*np.sin(psiDesired)
        
        e2 = wrapToPi(psi - psiDesired)
        e1dot = xdot * np.sin(e2) + ydot * np.cos(e2)
        e2dot = psidot

        longit_vel = xdot * np.cos(e2) - ydot * np.sin(e2)
        self._ConsoleUpdate(e1, np.degrees(e2), longit_vel)
        obs = np.array([longit_vel, e1, e1dot, e2, e2dot])
        

        # 2. compute reward
        r = self._get_reward(longit_vel, e2, np.abs(e1))
        # cost = self._get_cost(np.abs(e1))

        self.lastNearIdx = nearIdx

        # 3. done
        done = False

        # - case 1: succeed, 
        nearGoal = nearIdx >= len(self.trajectory) - 50
        if nearGoal and self.passMiddlePoint:
            done = True
        # - case 2: failed, too far from central line 
        if disError > 10.0:
            done = True
        # - case 3: failed, stuck somewhere
        if self.steps >= 1000:
            done = True
        
        # 4. info
        info = {}
        if nearGoal and self.passMiddlePoint:
            info['completed'] = True
        # info['cost'] = cost

        info_lst = [delT, X, Y, xdot, ydot, psi, psidot, f*15736, delta, disError]

        return obs, r, done, info, info_lst



if __name__ == "__main__":
    
    env = TracingEnv()
    episode_num = 1

    for episode in range(episode_num):
        obs = env.reset()
        print('obs',obs)

        count = 0
        done = False
        for _ in range(200):

            obs, reward, done, info = env.step([1000, -0.1])
            count += 1
            if count % 10 == 0:
                print('obs',obs)
                print('r',reward)
                print('--------------')
            if count >= 50:
                print('Test done.')
                break
        
        env.reset()
    # env.close()
