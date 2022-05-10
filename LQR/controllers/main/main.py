from controller import Display
from vehicle import Driver
from util import *
from lqr_controller import CustomController

trajectory = getTrajectory('buggyTrace.csv')

driver = Driver()
driver.setDippedBeams(True)
driver.setGear(1)
throttleConversion = 15737
timestep = int(driver.getBasicTimeStep())

desired_velocity = 12
customController = CustomController(trajectory, desired_velocity)
customController.startSensors(timestep)

XVec = []
YVec = []
deltaVec = []
xdotVec = []
ydotVec = []
psiVec = []
psidotVec = []
FVec = []
minDist = []
totalResult = []
passMiddlePoint = False
nearGoal = False
finish = False

while driver.step() != -1:

    X, Y, xdot, ydot, psi, psidot, F, delta = \
    customController.update(timestep)

    driver.setThrottle(clamp(F/throttleConversion, 0, 1))
    driver.setSteeringAngle(-clamp(delta, np.radians(-30), np.radians(30)))
    
    # Check for halfway point/completion
    disError, nearIdx = closestNode(X, Y, trajectory)
    stepToMiddle = nearIdx - len(trajectory)/2.0
    if abs(stepToMiddle) < 100.0 and passMiddlePoint == False:
        passMiddlePoint = True
        
    nearGoal = nearIdx >= len(trajectory) - 50
    if nearGoal and passMiddlePoint:
        finalPosition = trajectory[-25]
        finish = True
        break
        
    XVec.append(X)
    YVec.append(Y)
    deltaVec.append(delta)
    xdotVec.append(xdot)
    ydotVec.append(ydot)
    psiVec.append(psi)
    psidotVec.append(psidot)
    FVec.append(F)
    minDist.append(disError)
    
driver.setCruisingSpeed(0)
driver.setSteeringAngle(0)
if finish:
    maxminDist, avgDist = showResult(trajectory, timestep, \
                                       XVec, YVec, deltaVec, xdotVec, ydotVec, \
                                       FVec, psiVec, psidotVec, minDist)
    saveResult(XVec, YVec, deltaVec, xdotVec, ydotVec, \
               psiVec, psidotVec, FVec, minDist, \
               maxminDist, avgDist, totalResult)
    
    
