from sipcc.sipcc_problem import *
from sipcc.objective import *
from klampt import *
import time
from sipcc.geometryopt import *
import numpy as np
from sipcc.sip import SemiInfiniteConstraintInterface
from klampt.math import se3,so3,vectorops
import math
from klampt.io.numpy_convert import to_numpy
from sklearn.neighbors import KDTree
import heapq
from sipcc.sipcc import optimizeSIPCC
      
def find_nearest_point(point,point_cloud_tree,point_cloud_array,numeber_in_surrounding):
    """
    Find k closest points of a point in a point cloud.   
    """
    dist, idx = point_cloud_tree.query(np.asarray([point]), k=numeber_in_surrounding)
    points = point_cloud_array[idx].tolist()[0]
    return points

def score_function(grasping_problem,point,linkgeom_idx,CoM,x,y,z):
    """
    Calculate the goodness of a point for satisfying the constraints in a grasping planning problem. 
    """
    if len(y) != 0:
        balance_residual = grasping_problem.xyz_eq.value(x,y,z)
        diff_force = balance_residual[0:3]
    else:
        diff_force  = [0,0,9.8]
        
    # Goodness for force balance
    a1 = vectorops.dot(diff_force,grasping_problem.complementarity.obj.normal(point))

    # Distance to the object
    dist = (grasping_problem.complementarity.robot.geometry[linkgeom_idx].distance(list(point)))[0]
    a2 = math.exp(-5*dist)
    
    return a1+a2

class RobotKinematicsCache:
    def __init__(self,robot,gridres=0.001,pcres=0.001):
        self.robot = robot
        self.geometry = []
        for i in range(robot.numLinks()):
            geom = robot.link(i).geometry()
            if geom.empty():
                self.geometry.append(None)
            else:
                self.geometry.append(PenetrationDepthGeometry(geom,gridres,pcres))
        self.dirty = True
    def set(self,q):
        """Updates the robot configuration and PenetrationDepthGeometry transforms"""
        if self.dirty:
            self.robot.setConfig(q)
            for i in range(self.robot.numLinks()):
                if self.geometry[i] is None: continue
                self.geometry[i].setTransform(self.robot.link(i).getTransform())
            self.dirty = False
    def clear(self):
        self.dirty = True
    
class ConstantObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = c."""
    def __init__(self,c):
        self.c = np.int64(c)
        self.variable = 'x'
    def dims(self):
        return np.int64(0)
    def value(self,x):
        return self.c
    def gradient(self,x):
        return np.zeros(len(x))
    def hessian_x(self,x,y,z):
        return np.int64(1)

class RobotConfigObjective(ObjectiveFunctionInterface):
    def __init__(self,robot,qdes,weight=0.1):
        self.robot = robot
        self.qdes = qdes
        self.weight = weight
        self.variable = 'x'
    def dims(self):
        return np.int64(0)
    def value(self,q):
        return self.weight*self.robot.distance(q,self.qdes)**2
    def gradient(self,x):
        return self.weight*2*self.robot.distance(q,self.qdes)
    def hessian_x(self,x,y,z):
        return np.int64(1)    
        
class SumOfForceObjectiveFunction(ObjectiveFunctionInterface):
    """Sum of all the forces."""
    def __init__(self,norm='L1',weight=0.01):
        self.norm = norm
        self.variable = 'z'
        self.weight = weight
    def dims(self):
        return np.int64(0)
    def value(self,z):
        if self.norm == 'L1':
            return self.weight*vectorops.norm_L1(z)
        elif self.norm == 'L2':
            return 0.5*self.weight*vectorops.norm_L2(z)
        elif self.norm == 'Linf':
            return self.weight*vectorops.norm_Linf(z)
        else:
            raise Exception("Only L1, L2 and Linf norms are accepted!")
    def gradient(self,z):
        if self.norm == 'L1':
            return self.weight*np.sign(z)
        elif self.norm == 'L2':
            return self.weight*np.asarray(z)
        elif self.norm == 'Linf':
            z_abs = [abs(number) for number in z]
            z_max = max(z_abs)
            z_max_idx = z_abs.index(z_max)
            result = np.zeros(len(z))
            result[z_max_idx] = np.sign(z[z_max_idx])
            return self.weight*np.asarray(result)
        else:
            raise Exception("Only L1, L2 and Linf norms are accepted!")
    def hessian_x(self,x,y,z):
        return np.int64(1)
                
class RobotCollisionConstraint(SemiInfiniteConstraintInterface):
    """
    Note: requires geometries to support distance queries
    Note: modifies the robot's config on each call.
    """
    def __init__(self,robot,obj,gridres=0.001,pcres=0.001,robotcache=None):
        self.robot = robotcache if robotcache is not None else RobotKinematicsCache(robot,gridres,pcres)
        self.obj = PenetrationDepthGeometry(obj,gridres,pcres)
    def dims(self):
        return 0
    def setx(self,q):
        self.robot.set(q)
    def clearx(self):
        self.robot.clear()
    def value(self,q,index_pt):
        link_idx,point = index_pt
        return self.robot.geometry[link_idx].distance(point)[0]
    def minvalue(self,q):
        """
        Returns the smallest distances between each link and the object
        """
        result = []
        for i,linkgeom in enumerate(self.robot.geometry):
            if linkgeom:
                dmin,p_obj,p_rob = self.obj.distance(self.robot.geometry[i])
                result.append([i,dmin,p_obj,p_rob])
        return result
    def df_dx(self,q,index_pt):
        self.setx(q)
        link_idx,pt_obj = index_pt
        dist,point,_ = self.robot.geometry[link_idx].distance(pt_obj)
        worlddir = self.robot.geometry[link_idx].normal(point)
        Jp = self.robot.robot.link(link_idx).getPositionJacobian(self.robot.robot.link(link_idx).getLocalPosition(point[0:3]))
        self.clearx()
        return np.dot(np.array(Jp).T,worlddir)
    def domain(self):
        return self.obj

class FrictionConstraint(InfiniteConstraint):
    """
    The friction cone is linearized as an m-pyramid, z_f1,...,z_fm are the friction force and z_n is the normal force.
    This constraint requires friction_coefficient*z_n-(z_f1+...+z_fm) >= 0 for every index point.
    z = [z_f1,...,z_fm,z_n]
    """
    def __init__(self,friction_coefficient=0.7):
        self.friction_coefficient = friction_coefficient
    def dims(self):
        return 0
    def value(self,x,y,z):
        n = len(y)
        m = len(z)//n
        result = np.array([-1]*(m-1)+[self.friction_coefficient]).dot(z)
        return result
    def df_dx(self,x,y,z):
        return np.zeros(len(x))
    def df_dz(self,x,y,z):
        n = len(y)
        m = len(z)//n
        return np.array([-1]*(m-1)+[self.friction_coefficient])

class Equilibrium3DConstraint(InfiniteAggregateConstraint):
    """
    This constraint requires that the object to be gripped is in both force and torque balance.
    """
    def __init__(self,env_obj,env_geom,gravity_coefficient=-9.8,gravity_direction=[0,0,1],mass=1):
        self.env_obj = env_obj
        self.env_geom = env_geom
        pointcloud = to_numpy(self.env_geom.pc.getPointCloud(),type="PointCloud")
        R = to_numpy(self.env_geom.pc.getCurrentTransform()[0],'Matrix3')
        t = to_numpy(self.env_geom.pc.getCurrentTransform()[1],'Vector3')
        pointcloud[:,:3] = np.dot(R,pointcloud.T[:3,:]).T+t   
        self.point_cloud_array = pointcloud[:,:3]
        self.point_cloud_tree = KDTree(self.point_cloud_array, leaf_size=100)
        self.gravity_coefficient = gravity_coefficient
        self.gravity_direction = gravity_direction
        self.mass = mass
    def dims(self,y,z):
        return 6
    def value(self,x,y,z):

        result = np.zeros(6)
        
        if len(y) == 0:
            result[0:3] += self.mass * self.gravity_coefficient * np.asarray(self.gravity_direction)
            return result
        
        n = len(y)
        m = len(z)//n 
        T = self.env_geom.getTransform()
        CoM = se3.apply(T,self.env_obj.getMass().getCom())
        
        for i in range(n):
            
            point = y[i][1]
            f_normal = z[m*(i+1)-1]
            f_friction = z[m*i:m*(i+1)-1]
            Normal_normal = self.env_geom.normal(point)
            n1 = so3.canonical(Normal_normal)[3:6]
            n2 = so3.canonical(Normal_normal)[6:9]
            Normal_friction = []
            for j in range(6):
                n_tmp = (math.cos((math.pi/3)*j)*np.array(n1) + math.sin((math.pi/3)*j)*np.array(n2)).tolist()
                Normal_friction.append(vectorops.unit(n_tmp))            
            Cross_Normal = list(vectorops.cross(vectorops.sub(point,CoM), Normal_normal))
            Cross_Normal_friction = []
            for j in range(6):
                cross_Normal_v = list(vectorops.cross(vectorops.sub(point,CoM), Normal_friction[j]))
                Cross_Normal_friction.append(cross_Normal_v)
            
            result[0:3] += np.asarray(Normal_normal)*f_normal + np.asarray(f_friction)@np.asarray(Normal_friction)
            result[3:6] += np.asarray(Cross_Normal)*f_normal + np.asarray(f_friction)@np.asarray(Cross_Normal_friction)
            
        result[0:3] += self.mass * self.gravity_coefficient * np.asarray(self.gravity_direction)

        return result
    def df_dx(self,x,y,z):
        return np.zeros((6,len(x)))    
    def df_dz(self,x,y,z):
        n = len(y)
        m = len(z)//n
        result = np.zeros((6,len(z)))
        T = self.env_geom.getTransform()
        CoM = se3.apply(T,self.env_obj.getMass().getCom())
        
        for i in range(n):
            
            point = y[i][1]
            Normal_normal = self.env_geom.normal(point)
            n1 = so3.canonical(Normal_normal)[3:6]
            n2 = so3.canonical(Normal_normal)[6:9]
            Normal_friction = []
            for j in range(6):
                n_tmp = (math.cos((math.pi/3)*j)*np.array(n1) + math.sin((math.pi/3)*j)*np.array(n2)).tolist()
                Normal_friction.append(vectorops.unit(n_tmp))            
            Cross_Normal = list(vectorops.cross(vectorops.sub(point,CoM), Normal_normal))
            Cross_Normal_friction = []
            for j in range(6):
                cross_Normal_v = list(vectorops.cross(vectorops.sub(point,CoM), Normal_friction[j]))
                Cross_Normal_friction.append(cross_Normal_v)
            
            result[0:3,i*m:(i+1)*m-1] = np.asarray(Normal_friction).T
            result[0:3,(i+1)*m-1:(i+1)*m] = np.asarray(Normal_normal).reshape((3,1))
            result[3:6,i*m:(i+1)*m-1] = np.asarray(Cross_Normal_friction).T
            result[3:6,(i+1)*m-1:(i+1)*m] = np.asarray(Cross_Normal).reshape((3,1))
                
        return result

def grasping_oracle(grasping_problem,x,y,z,IndexSet,IndexSet_next,score_surrounding_size=500,sample_surrounding_size=50,sample_number=3,smallest_distance=0.001,score_oracle=False): 
       
    dim_robo = len(x)
    # Select new index point
    for i in range(dim_robo):
        if not score_oracle:
            if (dim_robo == 63 and i < 51) or dim_robo != 63: # i < 51: no index point on the legs of the humanoid
                if grasping_problem.complementarity.robot.geometry[i]: 
                    # Find the closest point between the ith link and the object
                    dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(grasping_problem.complementarity.obj)
                    # Only add the index points which are not too close to the existing index points                
                    if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in IndexSet):
                        IndexSet.append([i,p_obj])
                    
                    # Sample some points around the closest point  
                    pc_sub = find_nearest_point(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
                    sampled = []
                    successed_sample = []
                    while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:    
                        idx = np.random.randint(sample_surrounding_size)
                        if idx not in sampled:    
                            point = pc_sub[idx]
                            if all (vectorops.distance(idx[1],point) > smallest_distance for idx in IndexSet):
                                IndexSet.append([i,point])
                                successed_sample.append(idx)
                            sampled.append(idx)
                    
        # Add the point with the highest score as index point
        else:
            T = grasping_problem.xyz_eq.env_geom.getTransform()
            CoM = se3.apply(T,grasping_problem.xyz_eq.env_obj.getMass().getCom())
            
            if (dim_robo == 63 and i < 51) or dim_robo != 63: # i < 51: no index point on the legs of the humanoid 
                
                if grasping_problem.complementarity.robot.geometry[i]: 
                    # Find the closest point between the ith link and the object
                    dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(grasping_problem.complementarity.obj)
                    # Only add the index points which are not too close to the existing index points                
                    if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in IndexSet):
                        IndexSet.append([i,p_obj])
    
                    # Sample some points around the closest point  
                    pc_sub = find_nearest_point(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
                    sampled = []
                    successed_sample = []
                    while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:      
                        idx = np.random.randint(sample_surrounding_size)
                        if idx not in sampled:  
                            point = pc_sub[idx]
                            if all (vectorops.distance(idx[1],point) > smallest_distance for idx in IndexSet):
                                IndexSet.append([i,point])
                                successed_sample.append(idx)
                            sampled.append(idx)

                    # Calculate score for points in the surrounding of the closest point
                    score = []
                    pc_sub_score = find_nearest_point(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,score_surrounding_size)

                    for point in pc_sub_score:
                        score.append(score_function(grasping_problem,point,i,CoM,x,y,z))

                    # Find the point has the highest score
                    index = heapq.nlargest(1, list(range(len(score))), score.__getitem__)   
                    highest_score_point = pc_sub_score[index[0]]
                    if all (vectorops.distance(idx[1],highest_score_point) > smallest_distance for idx in IndexSet):
                        IndexSet.append([i,highest_score_point])
                    
                    # Sample some points around the highest score point
                    pc_sub = find_nearest_point(highest_score_point,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
                    sampled = []
                    successed_sample = []
                    while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:     
                        idx_ = np.random.randint(sample_surrounding_size)
                        if idx_ not in sampled:  
                            point = pc_sub[idx_]
                            dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(point)
                            if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in IndexSet):
                                IndexSet.append([i,p_obj])
                                successed_sample.append(idx)
                            sampled.append(idx_)
                    
    # Add penetration points detected in the previous iteration            
    if len(IndexSet_next) > 0:
        for index_point in IndexSet_next:
            link_idx,p_obj = index_point
            if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in IndexSet):
                IndexSet.append([link_idx,p_obj])
    
    print("===========================================")
    print("Oracle:")
    print(f"{len(IndexSet)} index points instantiated")
    
    return IndexSet
    
def optimizeGrasping(robot,objs,init_config,gridres,pcres,score_oracle=False):           

    robot.setConfig(init_config)
    
    # Set up collison constraints
    collision_constraints,pairs = makeCollisionConstraints(robot,objs,gridres,pcres)
    print(f"Created {len(collision_constraints)} collision constraints")
                
    q_init = robot.getConfig()  
    
    grasping_problem = SIPCCProblem()
    grasping_problem.set_objective(ConstantObjectiveFunction(0))
    grasping_problem.set_complementarity(RobotCollisionConstraint(robot,PenetrationDepthGeometry(objs[0].geometry(),gridres,pcres),gridres,pcres))
    grasping_problem.add_ineq(FrictionConstraint())
    grasping_problem.add_eq(Equilibrium3DConstraint(objs[0],PenetrationDepthGeometry(objs[0].geometry(),gridres,pcres)))
    grasping_problem.oracle = grasping_oracle
    
    x_init = q_init
    x_lb = grasping_problem.complementarity.robot.robot.getJointLimits()[0]
    x_ub = grasping_problem.complementarity.robot.robot.getJointLimits()[1]
    dim_z = 7
    
    x,y,z = optimizeSIPCC(grasping_problem,x_init,x_lb,x_ub,dim_z,score_oracle)
    
    return x,y,z