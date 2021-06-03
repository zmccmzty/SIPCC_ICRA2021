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

#ORACLE = 'MVO'
ORACLE = 'LSO'
#ORACLE = 'GSO'

NUM_FRICTION_CONE_EDGES = 6
LOCAL_NEIGHBORHOOD_SIZE = 100
EXTRA_SAMPLE_NEIGHBORHOOD_SIZE = 50
#EXTRA_SAMPLE_COUNT = 3
EXTRA_SAMPLE_COUNT = 0
MINIMUM_INDEX_DISTANCE = 0.001

class LinkAndGeometryDomain:
    """Index point domain is all points (link,pt), with link a robot link (int)
    and pt an environment point  (3-tuple).
    """
    def distance(self,y1,y2):
        if y1[0] != y2[0]: return np.inf
        return vectorops.distance(y1[1],y2[1])

_normal_cache = {}

def normal(env_geom,point):
    global _normal_cache
    point = tuple(point)
    if point in _normal_cache:
        return _normal_cache[point]
    n = env_geom.normal(point)
    _normal_cache[point] = n
    return n

def find_k_nearest(point,point_cloud_tree,point_cloud_array,k):
    """
    Find k closest points of a point in a point cloud.   
    """
    dist, idx = point_cloud_tree.query(np.asarray([point]), k=k)
    points = point_cloud_array[idx].tolist()[0]
    return points

def score_function(grasping_problem,point,linkgeom_idx,CoM,balance_residual,x,y,z):
    """
    Calculate the goodness of a point for satisfying the constraints in a grasping planning problem. 
    """
        
    # Goodness for force balance -- sort of like a cosine similarity
    a1 = vectorops.dot(balance_residual[:3],normal(grasping_problem.complementarity.obj,point))

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
    def value(self,x):
        return self.c
    def gradient(self,x):
        return np.zeros(len(x))
    def hessian(self,x):
        return np.int64(1)

class RobotConfigObjective(ObjectiveFunctionInterface):
    def __init__(self,robot,qdes,weight=0.1):
        self.robot = robot
        self.qdes = qdes
        self.weight = weight
    def value(self,q):
        return self.weight*self.robot.distance(q,self.qdes)**2
    def gradient(self,x):
        dq = (np.asarray(q)-self.qdes)  #TODO: non-Cartesian spaces?
        return self.weight*2*self.robot.distance(q,self.qdes)*dq
    def hessian(self,x):
        dq = (np.asarray(q)-self.qdes)  #TODO: non-Cartesian spaces?
        return self.weight*2*np.outer(dq,dq)
        
class SumOfForceObjectiveFunction(InfiniteObjectiveFunction):
    """Sum of all the forces."""
    def __init__(self,norm='L1',weight=0.01):
        self.norm = norm
        self.weight = weight
    def dims(self):
        return np.int64(0)
    def value(self,x,y,z):
        if self.norm == 'L1':
            return self.weight*vectorops.norm_L1(z)
        elif self.norm == 'L2':
            return 0.5*self.weight*vectorops.norm_L2(z)
        elif self.norm == 'Linf':
            return self.weight*vectorops.norm_Linf(z)
        else:
            raise Exception("Only L1, L2 and Linf norms are accepted!")
    def gradient_x(self,x,y,z):
        return np.zeros(x.shape)
    def gradient_z(self,x,y,z):
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
        return np.zeros((len(x),len(x)))
    def hessian_z(self,x,y,z):
        raise NotImplementedError("TODO")
        return np.zeros((len(x),len(x)))
                
class RobotCollisionConstraint(SemiInfiniteConstraintInterface):
    """
    Note: requires geometries to support distance queries
    Note: modifies the robot's config on each call.
    """
    def __init__(self,robot,obj,gridres=0.001,pcres=0.001,robotcache=None):
        self.robot = robotcache if robotcache is not None else RobotKinematicsCache(robot,gridres,pcres)
        self.obj = PenetrationDepthGeometry(obj,gridres,pcres)
        self.scale = 1
    def dims(self):
        return 0
    def setx(self,q):
        self.robot.set(q)
    def clearx(self):
        self.robot.clear()
    def value(self,q,index_pt):
        assert list(q) == self.robot.robot.getConfig()
        link_idx,point = index_pt
        return self.robot.geometry[link_idx].distance(point)[0]*self.scale
    def minvalue(self,q,bound=None):
        """
        Returns the smallest distance between each link and the object
        """
        # dmin = float('inf')
        # closest = None
        # if bound is None:
        #     bound = float('inf')
        closepts = []
        for i,linkgeom in enumerate(self.robot.geometry):
            if linkgeom:
                d,p_obj,p_rob = self.obj.distance(linkgeom,bound)
                #d,p_rob,p_obj = linkgeom.distance(self.obj,bound)
                closepts.append((d*self.scale,[i,p_obj]))
        return closepts
        #         if d < bound:
        #             bound = d
        #             closest = [i,p_obj]
        # return bound,closest
    def df_dx(self,q,index_pt):
        link_idx,pt_obj = index_pt
        dist,point,_ = self.robot.geometry[link_idx].distance(pt_obj)
        if dist > 1e-3:
            worlddir = vectorops.unit(vectorops.sub(point,pt_obj))
            #worlddir2 = self.robot.geometry[link_idx].normal(point)
            #worlddir3 = vectorops.mul(normal(self.obj,pt_obj),-1.0)
            #print("Comparing normals: {} vs {} vs {}".format('%.2f %.2f %.2f'%tuple(worlddir),'%.2f %.2f %.2f'%tuple(worlddir2),'%.2f %.2f %.2f'%tuple(worlddir3)))
        else:
            worlddir = self.robot.geometry[link_idx].normal(point)
            #worlddir = vectorops.mul(normal(self.obj,pt_obj),-1.0)
        Jp = self.robot.robot.link(link_idx).getPositionJacobian(self.robot.robot.link(link_idx).getLocalPosition(point[0:3]))
        return np.dot(np.array(Jp).T,worlddir)*self.scale
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
        assert len(z)==7
        result = np.array([-1]*6+[self.friction_coefficient]).dot(z)
        return result
    def df_dx(self,x,y,z):
        return np.zeros(len(x))
    def df_dz(self,x,y,z):
        assert len(z)==7
        return np.array([-1]*6+[self.friction_coefficient])

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
        assert m == 7
        T = self.env_geom.getTransform()
        CoM = se3.apply(T,self.env_obj.getMass().getCom())
        
        for i in range(n):
            point = y[i][1]
            f_normal = z[m*(i+1)-1]
            f_friction = z[m*i:m*(i+1)-1]
            Normal_normal = normal(self.env_geom,point)
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
            Normal_normal = normal(self.env_geom,point)
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

def maximum_violation_oracle(grasping_problem,x,y,z,
    sample_surrounding_size=EXTRA_SAMPLE_NEIGHBORHOOD_SIZE,sample_number=EXTRA_SAMPLE_COUNT,smallest_distance=MINIMUM_INDEX_DISTANCE): 
    IndexSet = []
    dim_robo = len(x)
    # Select new index point
    for i in range(dim_robo):
        if grasping_problem.colliding_links is not None and i not in grasping_problem.colliding_links:
            continue

        if not grasping_problem.complementarity.robot.geometry[i]: 
            continue

        # Find the closest point between the ith link and the object
        dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(grasping_problem.complementarity.obj)
        # Only add the index points which are not too close to the existing index points                
        IndexSet.append([i,p_obj])
        
        # Sample some points around the closest point  
        if sample_number > 0:
            pc_sub = find_k_nearest(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
            sampled = []
            successed_sample = []
            while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:    
                idx = np.random.randint(sample_surrounding_size)
                if idx not in sampled:    
                    point = pc_sub[idx]
                    if all (vectorops.distance(idx[1],point) > smallest_distance for idx in y + IndexSet):
                        IndexSet.append([i,point])
                        successed_sample.append(idx)
                    sampled.append(idx)

    return IndexSet


def local_score_oracle(grasping_problem,x,y,z,
    score_surrounding_size=LOCAL_NEIGHBORHOOD_SIZE,sample_surrounding_size=EXTRA_SAMPLE_NEIGHBORHOOD_SIZE,sample_number=EXTRA_SAMPLE_COUNT,smallest_distance=MINIMUM_INDEX_DISTANCE): 
    t0 = time.time()
    IndexSet = []
    dim_robo = len(x)
    T = grasping_problem.xyz_eq.env_geom.getTransform()
    CoM = se3.apply(T,grasping_problem.xyz_eq.env_obj.getMass().getCom())
    if len(y) != 0:
        balance_residual = grasping_problem.xyz_eq.value(x,y,z)
    else:
        balance_residual  = [0,0,9.8,0,0,0]
    # Select new index point
    for i in range(dim_robo):
        # Add the point with the highest score as index point
        if grasping_problem.colliding_links is not None and i not in grasping_problem.colliding_links:
            continue
                
        if not grasping_problem.complementarity.robot.geometry[i]: 
            continue

        # Find the closest point between the ith link and the object
        dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(grasping_problem.complementarity.obj)
        # Only add the index points which are not too close to the existing index points                
        if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in y):
            IndexSet.append([i,p_obj])

        # Sample some points around the closest point  
        if sample_number > 0:
            pc_sub = find_k_nearest(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
            sampled = []
            successed_sample = []
            while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:      
                idx = np.random.randint(sample_surrounding_size)
                if idx not in sampled:  
                    point = pc_sub[idx]
                    if all (vectorops.distance(idx[1],point) > smallest_distance for idx in y + IndexSet):
                        IndexSet.append([i,point])
                        successed_sample.append(idx)
                    sampled.append(idx)

        # Calculate score for points in the surrounding of the closest point
        score = []
        pc_sub_score = find_k_nearest(p_obj,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,score_surrounding_size)

        for point in pc_sub_score:
            score.append(score_function(grasping_problem,point,i,CoM,balance_residual,x,y,z))

        # Find the point has the highest score
        index = heapq.nlargest(1, list(range(len(score))), score.__getitem__)   
        highest_score_point = pc_sub_score[index[0]]
        if all (vectorops.distance(idx[1],highest_score_point) > smallest_distance for idx in y + IndexSet):
            IndexSet.append([i,highest_score_point])
        
        # Sample some points around the highest score point
        # pc_sub = find_k_nearest(highest_score_point,grasping_problem.xyz_eq.point_cloud_tree,grasping_problem.xyz_eq.point_cloud_array,sample_surrounding_size)
        # sampled = []
        # successed_sample = []
        # while len(successed_sample) < sample_number and len(sampled) < sample_surrounding_size:     
        #     idx_ = np.random.randint(sample_surrounding_size)
        #     if idx_ not in sampled:  
        #         point = pc_sub[idx_]
        #         dist, p_robo, p_obj = grasping_problem.complementarity.robot.geometry[i].distance(point)
        #         if all (vectorops.distance(idx[1],p_obj) > smallest_distance for idx in y + IndexSet):
        #             IndexSet.append([i,p_obj])
        #             successed_sample.append(idx)
        #         sampled.append(idx_)
                    
    t1 = time.time()
    print("Oracle computation time %.3f"%(t1-t0))
    return IndexSet
    
def optimizeGrasping(robot,objs,init_config,gridres,pcres,score_oracle=ORACLE,collision_links=None):

    robot.setConfig(init_config)
    
    robot_geometry = RobotKinematicsCache(robot,gridres,pcres)
    object_geometry = PenetrationDepthGeometry(objs[0].geometry(),gridres,pcres)
                
    q_init = robot.getConfig()  
    
    grasping_problem = SIPCCProblem()
    grasping_problem.dim_z = 7
    grasping_problem.z_proj = np.array([0,0,0,0,0,0,1])
    grasping_problem.z_lb = np.array([0,0,0,0,0,0,0])
    grasping_problem.domain = LinkAndGeometryDomain()
    grasping_problem.set_objective(ConstantObjectiveFunction(0))
    grasping_problem.set_complementarity(RobotCollisionConstraint(robot,object_geometry,gridres,pcres,robot_geometry))
    grasping_problem.add_ineq(FrictionConstraint())
    grasping_problem.add_eq(Equilibrium3DConstraint(objs[0],object_geometry))
    grasping_problem.colliding_links = collision_links
    if score_oracle=='LSO':
        grasping_problem.oracle = local_score_oracle
    elif score_oracle=='MVO':
        grasping_problem.oracle = maximum_violation_oracle
    elif score_oracle=='GSO':
        grasping_problem.oracle = global_score_oracle
    else:
        raise ValueError("score_oracle must be LSO, GSO, or MVO")
    
    x_init = q_init
    x_lb = grasping_problem.complementarity.robot.robot.getJointLimits()[0]
    x_ub = grasping_problem.complementarity.robot.robot.getJointLimits()[1]
    dim_z = 7
    
    from klampt import vis
    wdisplay = WorldModel()
    wdisplay.add('robot',robot)
    display_robot = wdisplay.robot(0)
    for i,o in enumerate(objs):
        wdisplay.add('obj_{}'.format(i),o)
    def update_robot_viz(x,y,z):
        vis.clear()
        vis.add("robot",display_robot)
        for i,o in enumerate(objs):
            odisplay = wdisplay.rigidObject(i)
            vis.add("obj_{}".format(i),odisplay)
        try:
            display_robot.setConfig(x)
            for i,yi in enumerate(y):
                vis.add("pt_{}".format(i),yi[1],color=(1,0,0,1),hide_label=True)
                f_normal = z[i*7+6]
                f_friction = z[i*7:i*7+6]
                n = np.array(normal(grasping_problem.xyz_eq.env_geom,yi[1]))
                f = n*f_normal
                n1 = so3.canonical(n)[3:6]
                n2 = so3.canonical(n)[6:9]
                for j in range(6):
                    n_tmp = (math.cos((math.pi/3)*j)*np.array(n1) + math.sin((math.pi/3)*j)*np.array(n2))
                    f += n_tmp*f_friction[j]
                #f is the force pointing inward
                #vis.add("n_{}".format(i),[yi[1],vectorops.madd(yi[1],normal,0.05)],color=(1,1,0,1),hide_label=True)
                vis.add("f_{}".format(i),[yi[1],vectorops.madd(yi[1],f.tolist(),-1.0)],color=(1,0.5,0,1),hide_label=True)
        finally:
            pass
        time.sleep(0.01)
    vis.show()
    settings = SIPCCOptimizationSettings()
    settings.callback = lambda *args:vis.threadCall(lambda : update_robot_viz(*args))
    settings.min_index_point_distance = MINIMUM_INDEX_DISTANCE
    x,y,z = optimizeSIPCC(grasping_problem,x_init,x_lb,x_ub,settings=settings)
    vis.spin(float('inf'))

    return x,y,z