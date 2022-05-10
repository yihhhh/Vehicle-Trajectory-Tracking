import numpy as np
import matplotlib.pyplot as plt
import csv

def wrapToPi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def clamp(n, minValue, maxValue):
    return max(min(maxValue, n), minValue)

def closestNode(X, Y, trajectory):
    point = np.array([X, Y])
    trajectory = np.asarray(trajectory)
    dist = point - trajectory
    distSquared = np.sum(dist ** 2, axis=1)
    minIndex = np.argmin(distSquared)
    return np.sqrt(distSquared[minIndex]), minIndex

def getTrajectory(filename):
    with open(filename) as f:
        lines = f.readlines()
        trajectory = np.zeros((len(lines), 2))
        for idx, line in enumerate(lines):
            x = line.split(",")
            trajectory[idx, 0] = x[0]
            trajectory[idx, 1] = x[1]
    return trajectory

def showResult(traj, timestep, X, Y, delta, xdot, ydot, F, psi, psidot, minDist):
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
    print('Maximum distance off reference trajectory: ', max(minDist))
    print('Average distance off reference trajectory: ', avgDist)
    plt.show()
    
    return max(minDist), avgDist

def saveResult(XVec, YVec, deltaVec, xdotVec, ydotVec, psiVec, psidotVec, FVec, minDist, maxminDist, avgDist, totalResults):
    totalResults.append(XVec)
    totalResults.append(YVec)
    totalResults.append(deltaVec)
    totalResults.append(xdotVec)
    totalResults.append(ydotVec)
    totalResults.append(psiVec)
    totalResults.append(psidotVec)
    totalResults.append(FVec)
    totalResults.append(minDist)
    # totalResults.append(maxminDist)
    # totalResults.append(avgDist)
    
    totalResults = np.array(totalResults)
    #print(totalResults)
    
    np.savetxt("lqr_result.csv", totalResults, delimiter = ",", fmt = "%1.4f")