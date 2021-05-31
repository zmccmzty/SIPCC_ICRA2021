from klampt.math import vectorops
from .mpcc import *
from .sipcc_problem import *
import numpy as np
import copy

def lineSearch(sipcc_problem,w_init,w_k,x_dim,y_k,f,c,w_lb,w_ub,c_lb,c_ub,LS_max_iters=30,LS_shrink_coef=0.8):
    
    IndexSet_next = []
    w_diff = vectorops.sub(w_k,w_init) 
    dim_robo = x_dim
    maximum_violation_weight = 1000
    
    # Calculate merit value before optimization              
    x_init = w_init[:dim_robo]
    sipcc_problem.complementarity.setx(x_init)
    maximum_violations_detail = sipcc_problem.complementarity.minvalue(x_init)
    maximum_violations = [abs(min(maximum_violation_detail[1],0)) for maximum_violation_detail in maximum_violations_detail]
    maximum_violation_before_opt = sum(maximum_violations)
    c_value = c.value(w_init)
    sipcc_problem.complementarity.clearx()
    
    merit_before_opt = 0
    merit_before_opt += f.value(w_init)
    merit_before_opt += vectorops.norm_L1(vectorops.minimum(vectorops.sub(c_ub,c_value),0))
    merit_before_opt += vectorops.norm_L1(vectorops.minimum(vectorops.sub(c_value,c_lb),0))
    merit_before_opt += vectorops.norm_L1(vectorops.minimum(vectorops.sub(w_ub,w_init),0))
    merit_before_opt += vectorops.norm_L1(vectorops.minimum(vectorops.sub(w_init,w_lb),0))
    merit_before_opt += maximum_violation_weight*maximum_violation_before_opt
        
    converged = False
    alpha = 1
    iteration = 0
    
    while (iteration < LS_max_iters) and (not converged):
                    
        w_tmp = vectorops.add(w_init,vectorops.mul(w_diff,alpha))
        
        x_tmp = w_tmp[:dim_robo]
        z_tmp = w_tmp[dim_robo:]
        sipcc_problem.complementarity.setx(x_tmp)
        maximum_violations_detail_tmp = sipcc_problem.complementarity.minvalue(x_tmp)
        maximum_violations_tmp = [abs(min(maximum_violation_detail[1],0)) for maximum_violation_detail in maximum_violations_detail_tmp]
        maximum_violation_tmp = sum(maximum_violations_tmp)
        c_value = c.value(w_tmp)
        sipcc_problem.complementarity.clearx()
        
        merit_tmp = 0
        merit_tmp += f.value(w_tmp)
        merit_tmp += vectorops.norm_L1(vectorops.minimum(vectorops.sub(c_ub,c_value),0))
        merit_tmp += vectorops.norm_L1(vectorops.minimum(vectorops.sub(c_value,c_lb),0))
        merit_tmp += vectorops.norm_L1(vectorops.minimum(vectorops.sub(w_ub,w_init),0))
        merit_tmp += vectorops.norm_L1(vectorops.minimum(vectorops.sub(w_init,w_lb),0))
        constraints_violation = merit_tmp - f.value(w_tmp) + maximum_violation_tmp
        merit_tmp += maximum_violation_weight*maximum_violation_tmp
                    
        if not merit_tmp < merit_before_opt:
            alpha = alpha * LS_shrink_coef
            iteration += 1
        else:
            converged = True
    
    print("Line Search:")
    print(f"Merit before opt: {merit_before_opt}")
    print(f"Merit after opt: {merit_tmp}")
    print(f"Alpha: {alpha}, Iteration: {iteration}")
    if len(IndexSet_next) > 0:
        print(f"{len(IndexSet_next)} points detected in line search! \n")
    
    complementarity_gap = vectorops.norm_L1(vectorops.minimum(vectorops.sub(c_ub[len(z_tmp):2*len(z_tmp)],c_value[len(z_tmp):2*len(z_tmp)]),0))
    total_maximum_violation = maximum_violation_tmp
    return w_tmp, IndexSet_next, total_maximum_violation, complementarity_gap, constraints_violation

def optimizeSIPCC(sipcc_problem,x_init,x_lb,x_ub,dim_z,score_oracle):
    settings = SIPCCOptimizationSettings()
    
    max_iters = settings.max_iters
    warm_start_z = settings.warm_start_z
    mpcc_solver_params = settings.mpcc_solver_params
    Major_step_limit = mpcc_solver_params['Major_step_limit']
    Max_Iters_init = mpcc_solver_params['Max_Iters_init']
    
    iters = 0
    x_k = x_init
    y_k = []
    z_k = []
    IndexSet = []
    IndexSet_next = []
    y_k_kept = []
    z_k_kept = []
    IndexSet_kept = []
    dim_x = len(x_k)
    total_maximum_violation = np.inf
    complementarity_gap = np.inf 

    while iters < max_iters and (iters == 0 or vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k)) > 0.02 \
                                 or total_maximum_violation > 0.001 or complementarity_gap > 0.001):
        
        IndexSet = sipcc_problem.oracle(sipcc_problem,x_k,y_k_kept,z_k_kept,IndexSet,IndexSet_next,score_oracle)
        IndexSet = IndexSet_kept + IndexSet
        IndexSet_next = []
        x_k_init = x_k.copy()
        y_k = IndexSet.copy()
        
        if not warm_start_z:
            z_k_init = [0]*len(IndexSet)*7
        else:
            z_k_init = z_k_kept + [0]*(len(IndexSet)-len(IndexSet_kept))*7
        complementarity_slack = max(min(0.1**(iters),0.001),1e-6)
        
        if sipcc_problem.complementarity:
            sipcc_problem.complementarity.IndexSet = IndexSet
        if sipcc_problem.z_ineq:
            sipcc_problem.z_ineq.IndexSet = IndexSet
        if sipcc_problem.z_eq:
            sipcc_problem.z_eq.IndexSet = IndexSet
        if sipcc_problem.xyz_ineq:
            sipcc_problem.xyz_ineq.IndexSet = IndexSet
        if  sipcc_problem.xyz_eq:
            sipcc_problem.xyz_eq.IndexSet = IndexSet
        
        f,c,w_lb,w_ub,c_lb,c_ub = sipcc_problem.getNLP(x_k_init,y_k,z_k_init,x_lb,x_ub,complementarity_slack)
        w_k_init = x_k_init + z_k_init
        res = optimizeMPCC(f,c,w_k_init,w_lb,w_ub,c_lb,c_ub,Major_step_limit,min(Max_Iters_init+2*iters,50))
        w_k = res.xStar["xvars"]
        
        # Line Search
        w_k,IndexSet_next,total_maximum_violation,complementarity_gap, constrains_violation = \
            lineSearch(sipcc_problem,w_k_init,w_k,dim_x,y_k,f,c,w_lb,w_ub,c_lb,c_ub)
        x_k = w_k[:len(x_k_init)]
        z_k = w_k[len(x_k_init):]
        
        # Delete Index Points
        IndexSet_tmp = []
        z_tmp = []
        for i in range(0,len(IndexSet)):
            if w_k[dim_x+dim_z*(i+1)-1] >= 0.05 and sipcc_problem.complementarity.value(x_k,IndexSet[i]) <= 0.005:
                IndexSet_tmp.append(IndexSet[i])
                z_tmp.append(w_k[dim_x+dim_z*i:dim_x+dim_z*(i+1)])
        
        IndexSet_kept = IndexSet_tmp   
        z_k_kept = list(np.concatenate(z_tmp)) if len(z_tmp) != 0 else []
        y_k_kept = copy.deepcopy(IndexSet_kept)
        IndexSet = copy.deepcopy(IndexSet_kept)
        
        print("Index Point Deletion (d > 0.005):")
        print(f"Points before deletion: {len(z_k)//dim_z}")
        print(f"Points after deletion: {len(IndexSet)}")  
        
        print(f"Iteration {iters} result:")                   
        print(f"Total Maximum Violation: {total_maximum_violation}")
        print(f"Complementarity Gap: {complementarity_gap}")
        if sipcc_problem.xyz_eq:
            print(f"InfiniteAggregateConstraint violation: {vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k))}")
        print("=========================================== \n")
    
        iters += 1
        
    return x_k,y_k,z_k