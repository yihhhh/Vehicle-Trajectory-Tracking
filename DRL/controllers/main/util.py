# Please do not change this file
# The file contains the helper functions one may need for the project

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

def wrapToPi(a):
    # Wraps the input angle to between 0 and pi
    return (a + np.pi) % (2 * np.pi) - np.pi

def clamp(n, minValue, maxValue):
    # Clamps to range [minValue, maxValue]
    return max(min(maxValue, n), minValue)

def closestNode(X, Y, trajectory):
    # Finds the closest waypoints in the trajectory
    # with respect to the point [X, Y] given in as input
    point = np.array([X, Y])
    trajectory = np.asarray(trajectory)
    dist = point - trajectory
    distSquared = np.sum(dist ** 2, axis=1)
    minIndex = np.argmin(distSquared)
    return np.sqrt(distSquared[minIndex]), minIndex

def getTrajectory(filename):
    # Import .csv file and stores the waypoints
    # in a 2D numpy array --> trajectory
    with open(filename) as f:
        lines = f.readlines()
        trajectory = np.zeros((len(lines), 2))
        for idx, line in enumerate(lines):
            x = line.split(",")
            trajectory[idx, 0] = x[0]
            trajectory[idx, 1] = x[1]
    return trajectory

class DisplayUpdate:
    def __init__(self, display):
        self.display = display

    def refresh(self):
        x = self.display.getWidth()
        y = self.display.getHeight()
        self.display.setAlpha(1.0)
        self.display.setColor(0x000000)
        self.display.fillRectangle(0, 0, x, y)
        self.display.setColor(0xFFFFFF)

    def consoleUpdate(self, disError, nearIdx):
        self.refresh()
        self.display.drawText("Cross-track error: " + str(round(disError,5)) + \
                         "\n\nNearest waypoint: " + str(nearIdx) + \
                         "\n\nPercent complete: " + str(round((nearIdx/8153)*100,1)) + "%", 5, 5)

    def speedometerUpdate(self, graphic, xdot):
        self.refresh()
        self.display.imagePaste(graphic, 0, 0, True)
        needleLength = 50
        alpha = xdot / 130.0 * 3.72 - 0.27
        x = int(-needleLength * np.cos(alpha))
        y = int(-needleLength * np.sin(alpha))
        self.display.drawLine(100, 95, 100 + x, 95 + y)

def showResult(traj, timestep, X, Y, delta, xdot, ydot, F, psi, psidot, minDist):
    # Function to plot the entire history of the car
    # Plot - X v/s Y
    # Plot - time (s) vs. delta (rad)
    # Plot - time (s) vs. xdot (m/s)
    # Plot - time (s) vs. ydot (m/s)
    # Plot - time (s) vs. psi (rad)
    # Plot - time (s) vs. psidot (rad/s)
    totalTime = np.linspace(0, len(X)*timestep*0.001, len(X))
    print('total steps: ', timestep*len(X))

    fig, _ = plt.subplots(nrows = 4, ncols = 2,figsize = (15,10))

    plt.subplot(421)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.plot(traj[:, 0], traj[:, 1], 'gray',linewidth=6.0)
    plt.plot(X, Y, 'r')

    plt.subplot(422)
    plt.xlabel('Time (s)')
    plt.ylabel('delta (rad)')
    plt.plot(totalTime[2:], delta[2:], 'r')

    plt.subplot(423)
    plt.xlabel('Time (s)')
    plt.ylabel('xdot (m/s)')
    plt.plot(totalTime[2:], xdot[2:], 'r')

    plt.subplot(424)
    plt.xlabel('Time (s)')
    plt.ylabel('ydot (m/s)')
    plt.plot(totalTime[2:], ydot[2:], 'r')

    plt.subplot(425)
    plt.xlabel('Time (s)')
    plt.ylabel('psi (rad)')
    plt.plot(totalTime[2:], psi[2:], 'r')

    plt.subplot(426)
    plt.xlabel('Time (s)')
    plt.ylabel('psidot (rad/s)')
    plt.plot(totalTime[2:], psidot[2:], 'r')

    plt.subplot(427)
    plt.xlabel('Time (s)')
    plt.ylabel('minDist (m)')
    plt.plot(totalTime[2:], minDist[2:], 'r')

    plt.subplot(428)
    plt.xlabel('Time (s)')
    plt.ylabel('F (N)')
    plt.plot(totalTime[2:], F[2:], 'r')

    fig.tight_layout()

    avgDist = sum(minDist) / len(minDist)
    print('maxMinDist: ', max(minDist))
    print('avgMinDist: ', avgDist)
    plt.show()
