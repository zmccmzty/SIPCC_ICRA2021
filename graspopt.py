from sipcc.sipcc_problem import *
from sipcc.objective import *
from klampt import *
from klampt import vis
import time
from sipcc.geometryopt import *
import sys
from sipcc.graspingopt import optimizeGrasping

testcase = 'sphere'
if len(sys.argv) > 1:
    testcase = str(sys.argv[1])
    print(testcase)
    
if testcase == 'sphere':   # Gripper & Sphere
    worldfilename = "data/grasping_test_sphere.xml"
    print("=======================")
    print("Gripper grasps a sphere")
    print("=======================")
elif testcase == 'glass': # Gripper & Wine glass
    worldfilename = "data/grasping_test_wineglass.xml"
    print("===========================")
    print("Gripper grasps a wine glass")
    print("===========================")
elif testcase == 'bowl': # Gripper & Bowl
    worldfilename = "data/grasping_test_bowl.xml" 
    print("=====================")
    print("Gripper grasps a bowl")
    print("=====================")    
elif testcase == 'hubo': # Humanoid & Sphere
    worldfilename = "data/Hubo_hold_object.xml"
    print("=====================")
    print("Humanoid grasps a sphere")
    print("=====================") 

if testcase == 'hubo': # Humanoid & Sphere
    gridres = 0.004
    pcres = 0.004
else:
    gridres = 0.0008
    pcres = 0.001 

# Set up the robot and object    
world = WorldModel()
world.readFile(worldfilename)
robot = world.robot(0)
obstacles = []
for i in range(world.numRigidObjects()):
    obstacles.append(world.rigidObject(i))
object_ = obstacles[0] 
print(f"{world.numRobots()} robots, {world.numRigidObjects()} rigid objects")

# Set initial configuretion for the robot
if testcase == 'sphere':
    init_config = [0.022, -0.0, 0.41, 0.0, 0.0, -3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
if testcase == 'glass':
    init_config = [0.1, 0.06, 0.67, -1.57, 0.0, -3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
if testcase == 'bowl':   
    init_config = [0.15, -0.25, 0.35, 1.57, 1.57, -3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
if testcase == 'hubo':  
    init_config[0] * len(robot.getConfig())
    init_config[0] = 0.05
    init_config[2] = -0.1
    init_config[8] = -1.14
    init_config[11] = -0.4
    init_config[12] = 0.52
    init_config[13] = 0.5
    init_config[17] = -1
    init_config[20] = -1
    init_config[23] = -1
    init_config[26] = -1
    init_config[29] = -1.14
    init_config[32] = -0.4
    init_config[33] = -0.52
    init_config[34] = 0.5
    init_config[38] = -1
    init_config[41] = -1
    init_config[44] = -1
    init_config[47] = -1

t_start = time.time()        
x,y,z = optimizeGrasping(robot,obstacles,init_config,gridres,pcres,score_oracle=True)
t_end = time.time()
print(f"Total time: {t_end-t_start}")
robot.setConfig(x)

vis.add("world",world)
vis.show() 
while vis.shown:
    vis.lock()
    vis.unlock()
    time.sleep(0.001)
vis.kill()