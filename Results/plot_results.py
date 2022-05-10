import numpy as np
import matplotlib.pyplot as plt

def getTrajectory(filename):
    with open(filename) as f:
        lines = f.readlines()
        trajectory = np.zeros((len(lines), 2))
        for idx, line in enumerate(lines):
            x = line.split(",")
            trajectory[idx, 0] = x[0]
            trajectory[idx, 1] = x[1]
    return trajectory


def showResult(lqr_result, ppo_result, traj):
    lqr_XVec = lqr_result[0, :]
    lqr_YVec = lqr_result[1, :]
    lqr_deltaVec = lqr_result[2, :]
    lqr_xdotVec = lqr_result[3, :]
    lqr_ydotVec = lqr_result[4, :]
    lqr_psiVec = lqr_result[5, :]
    lqr_psidotVec = lqr_result[6, :]
    lqr_FVec = lqr_result[7, :]
    lqr_minDist = lqr_result[8, :]
    
    ppo_XVec = ppo_result[0, :]
    ppo_YVec = ppo_result[1, :]
    ppo_deltaVec = ppo_result[2, :]
    ppo_xdotVec = ppo_result[3, :]
    ppo_ydotVec = ppo_result[4, :]
    ppo_FVec = ppo_result[5, :]
    ppo_psiVec = ppo_result[6, :]
    ppo_psidotVec = ppo_result[7, :]
    ppo_minDist = ppo_result[8, :]
    
    timestep = 32
    lqr_totalTime = np.linspace(0, len(lqr_XVec) * timestep * 0.001, len(lqr_XVec))
    ppo_totalTime = np.linspace(0, len(ppo_XVec) * timestep * 0.005, len(ppo_XVec))

    fig, _ = plt.subplots(nrows = 4, ncols = 2,figsize = (15,10))

    plt.subplot(421)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.plot(traj[:, 0], traj[:, 1], 'gray',linewidth=6.0)
    plt.plot(lqr_XVec, lqr_YVec, 'r')
    plt.plot(ppo_XVec, ppo_YVec, 'b')

    plt.subplot(422)
    plt.xlabel('Time (s)')
    plt.ylabel('delta (rad)')
    plt.plot(lqr_totalTime[2:], lqr_deltaVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_deltaVec[2:], 'b')

    plt.subplot(423)
    plt.xlabel('Time (s)')
    plt.ylabel('xdot (m/s)')
    plt.plot(lqr_totalTime[2:], lqr_xdotVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_xdotVec[2:], 'b')

    plt.subplot(424)
    plt.xlabel('Time (s)')
    plt.ylabel('ydot (m/s)')
    plt.plot(lqr_totalTime[2:], lqr_ydotVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_ydotVec[2:], 'b')

    plt.subplot(425)
    plt.xlabel('Time (s)')
    plt.ylabel('psi (rad)')
    plt.plot(lqr_totalTime[2:], lqr_psiVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_psiVec[2:], 'b')

    plt.subplot(426)
    plt.xlabel('Time (s)')
    plt.ylabel('psidot (rad/s)')
    plt.plot(lqr_totalTime[2:], lqr_psidotVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_psidotVec[2:], 'b')

    plt.subplot(427)
    plt.xlabel('Time (s)')
    plt.ylabel('minDist (m)')
    plt.plot(lqr_totalTime[2:], lqr_minDist[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_minDist[2:], 'b')

    plt.subplot(428)
    plt.xlabel('Time (s)')
    plt.ylabel('F (N)')
    plt.plot(lqr_totalTime[2:], lqr_FVec[2:], 'r')
    plt.plot(ppo_totalTime[2:], ppo_FVec[2:], 'b')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    lqr_result = np.genfromtxt("lqr_result.csv", delimiter = ",")
    ppo_result = np.genfromtxt("ppo_result.csv", delimiter = ",")
    traj = getTrajectory('buggyTrace.csv')
    
    showResult(lqr_result, ppo_result, traj)


