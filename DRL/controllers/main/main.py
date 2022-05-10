# This file should be set as the controller for the Tesla robot node.
# Please do not alter this file - it may cause the simulation to fail.

# Import Webots-specific functions
from controller import Display
from vehicle import Driver

# Import functions from other scripts in controller folder
from util import *
from your_controller_pid_sol import CustomController
from evaluation import evaluation

trajectory = getTrajectory('buggyTrace.csv')

# Instantiate supervisor and functions
driver = Driver()
driver.setDippedBeams(True)
driver.setGear(1) # Torque control mode
throttleConversion = 15737
msToKmh = 3.6

# Access and set up displays
console = driver.getDisplay("console")
speedometer = driver.getDisplay("speedometer")
console.setFont("Arial Black", 14, True)
speedometerGraphic = speedometer.imageLoad("speedometer.png")
speedometer.imagePaste(speedometerGraphic, 0, 0, True)

consoleObject = DisplayUpdate(console)
speedometerObject = DisplayUpdate(speedometer)

# Get the time step of the current world
timestep = int(driver.getBasicTimeStep())

# Instantiate controller and start sensors
customController = CustomController(trajectory)
customController.startSensors(timestep)

# Initialize state storage vectors and completion conditions
XVec = []
YVec = []
deltaVec = []
xdotVec = []
ydotVec = []
psiVec = []
psidotVec = []
FVec = []
minDist = []
passMiddlePoint = False
nearGoal = False
finish = False

while driver.step() != -1:

    # Call control update method
    X, Y, xdot, ydot, psi, psidot, F, delta = \
    customController.update(timestep)

    # Set control update output
    driver.setThrottle(clamp(F/throttleConversion, 0, 1))
    driver.setSteeringAngle(-clamp(delta, np.radians(-30), np.radians(30)))
    
    # Check for halfway point/completion
    disError, nearIdx = closestNode(X, Y, trajectory)
    
    consoleObject.consoleUpdate(disError, nearIdx)
    speedometerObject.speedometerUpdate(speedometerGraphic, xdot*msToKmh)

    stepToMiddle = nearIdx - len(trajectory)/2.0
    if abs(stepToMiddle) < 100.0 and passMiddlePoint == False:
        passMiddlePoint = True
        
    if passMiddlePoint == True:
        console.drawText("Middle point passed.", 5, 60)
        
    nearGoal = nearIdx >= len(trajectory) - 50
    if nearGoal and passMiddlePoint:
        console.drawText("Middle point passed.\n\nDestination reached! :)", 5, 60)
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
    
# Reset position and physics once loop is completed,
# and print evaluation to console
driver.setCruisingSpeed(0)
driver.setSteeringAngle(0)
if finish:
    evaluation(minDist, trajectory, XVec, YVec)
    showResult(trajectory, timestep, \
               XVec, YVec, deltaVec, xdotVec, ydotVec, \
               FVec, psiVec, psidotVec, minDist)
