from klampt.math import vectorops
from .mpcc import *
from .sipcc_problem import *
import numpy as np
import copy
import time

def lineSearch(sipcc_problem,w_init,w_k,x_dim,y_k,f,c,w_lb,w_ub,c_lb,c_ub,weight_f,settings):
    """Descends a merit function::

        :math:`weight_f*f(w) - \min(d(w),0) + \phi(g*(x))`
    
    with:

    - d(w) >= 0 the constraint merging w_lb <= w <= w_lb and c_lb <= c(w) <= c_lb,
    - g*(x) = min_{y in D} g(x,y),
    - \phi(v) = inf*I[v > max_violation_limit] + max_violation_merit_weight*-min(v,0)

    """
    IndexSet_next = []
    w_diff = w_k-w_init
    dim_robo = x_dim
    
    # Calculate merit value before optimization              
    x_init = w_init[:dim_robo]
    sipcc_problem.complementarity.setx(x_init)
    minvalue_res = sipcc_problem.complementarity.minvalue(x_init)
    maximum_violations = [r[0] for r in minvalue_res]
    maximum_violation_pts = [r[1] for r in minvalue_res]
    maximum_violation_before_opt = np.sum(np.abs(np.minimum(maximum_violations,0)))
    maximum_violation_limit = max(settings.max_violation_limit,maximum_violation_before_opt*1.1)
    sipcc_problem.complementarity.clearx()
    c_value = c(w_init)
    
    merit_before_opt = 0
    merit_before_opt += f.value(w_init)
    merit_before_opt += -np.sum(np.minimum(c_ub-c_value,0))
    merit_before_opt += -np.sum(np.minimum(c_value-c_lb,0))
    merit_before_opt += -np.sum(np.minimum(w_ub-w_init,0))
    merit_before_opt += -np.sum(np.minimum(w_init-w_lb,0))
    merit_before_opt += settings.max_violation_merit_weight*maximum_violation_before_opt
        
    converged = False
    alpha = 1
    iteration = 0
    
    while (iteration < settings.LS_max_iters) and (not converged):
        w_tmp = w_init + w_diff*alpha
        w_tmp[dim_robo:] = w_k[dim_robo:]
        
        x_tmp = w_tmp[:dim_robo]
        z_tmp = w_tmp[dim_robo:]
        
        c_value = c(w_tmp)

        fw = f.value(w_tmp)        
        merit_tmp = 0
        merit_tmp += fw
        merit_tmp += -np.sum(np.minimum(c_ub-c_value,0))
        merit_tmp += -np.sum(np.minimum(c_value-c_lb,0))
        merit_tmp += -np.sum(np.minimum(w_ub-w_init,0))
        merit_tmp += -np.sum(np.minimum(w_init-w_lb,0))
        maximum_violation_tmp = 0
        if merit_tmp < merit_before_opt or alpha == 1:
            sipcc_problem.complementarity.setx(x_tmp)
            minvalue_res = sipcc_problem.complementarity.minvalue(x_tmp)
            maximum_violations = [r[0] for r in minvalue_res]
            maximum_violation_pts = [r[1] for r in minvalue_res]
            maximum_violation_tmp = np.sum(np.abs(np.minimum(maximum_violations,0)))
            sipcc_problem.complementarity.clearx()
            merit_tmp += settings.max_violation_merit_weight*maximum_violation_tmp
            
        if maximum_violation_tmp > maximum_violation_limit or not (merit_tmp < merit_before_opt):
            if alpha==1:
                print(f"Full-step merit value: {merit_tmp}, penetration {maximum_violation_tmp}")
            if maximum_violation_tmp != 0:
                IndexSet_next = []
                for d,y in zip(maximum_violations,maximum_violation_pts):
                    if d > settings.max_violation_limit*0.25:
                        IndexSet_next.append(y)
            alpha = alpha * settings.LS_shrink_coef
            iteration += 1
        else:
            converged = True
            if alpha==1:
                print(f"Full-step merit value: {merit_tmp}, penetration {maximum_violation_tmp}")
    
    print("Line Search:")
    print(f"  Merit before opt: {merit_before_opt}")
    print(f"  Merit after opt: {merit_tmp}")
    print(f"  Alpha: {alpha}, Iteration: {iteration}")
    if not converged:
        w_tmp = w_init
        IndexSet_next = []
        maximum_violation_tmp = maximum_violation_before_opt
    if len(IndexSet_next) > 0:
        print(f"  {len(IndexSet_next)} points detected in line search! ")
    
    total_maximum_violation = maximum_violation_tmp
    return converged, w_tmp, IndexSet_next, total_maximum_violation

def optimizeSIPCC(sipcc_problem,x_init,x_lb,x_ub,settings=SIPCCOptimizationSettings()):
    mpcc_solver_params = settings.mpcc_solver_params
    Solver = mpcc_solver_params['Solver']
    Major_step_limit = mpcc_solver_params['Major_step_limit']
    Max_Iters_init = mpcc_solver_params['Max_Iters_init']
    Max_Iters_growth = mpcc_solver_params['Max_Iters_growth']
    dim_z = sipcc_problem.dim_z

    iters = 0
    x_k = x_init
    y_k = []
    z_k = []
    IndexSet = []
    dim_x = len(x_k)
    total_maximum_violation = np.inf
    complementarity_gap = np.inf 

    while iters < settings.max_iters and (iters == 0 or vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k)) > settings.equality_tolerance \
                                 or total_maximum_violation > settings.constraint_violation_tolerance or complementarity_gap > settings.complementarity_gap_tolerance):
        
        if settings.callback:
            settings.callback(x_k,y_k,z_k)

        IndexSet_last = copy.deepcopy(IndexSet)
        x_k_last = x_k.copy()
        y_k_last = copy.deepcopy(y_k)
        z_k_last = copy.deepcopy(z_k)
        IndexSet_new = sipcc_problem.oracle(sipcc_problem,x_k,y_k,z_k)
        
        num_added = 0
        for i in IndexSet_new:
            if not any(sipcc_problem.domain.distance(i,ind) < settings.min_index_point_distance for ind in IndexSet):
                IndexSet.append(i)
                num_added += 1

        print("===========================================")
        print("Oracle:")
        print(f"{len(IndexSet_new)} index points returned, {num_added} added")
        y_k = IndexSet

        #instantiate Z variables at 0
        z_k_init = z_k_last + [0]*(dim_z*num_added)
        if not settings.warm_start_z:
            z_k_init = [0]*len(z_k_init)
        
        if settings.callback:
            settings.callback(x_k,y_k,z_k_init)
            
        #complementarity_slack = max(min(0.1**(iters),0.001),1e-6)
        complementarity_slack = max(0.01*(0.2**iters),1e-7)
        print("COMPLEMENTARITY SLACK",complementarity_slack)
        
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
        
        f,c,w_lb,w_ub,c_lb,c_ub = sipcc_problem.getNLP(x_k,y_k,z_k_init,x_lb,x_ub,complementarity_slack)
        w_k_init = np.hstack([x_k,z_k_init])
        res = optimizeMPCC(f,c,w_k_init,w_lb,w_ub,c_lb,c_ub,Solver,Major_step_limit,min(Max_Iters_init+Max_Iters_growth*iters,50))
        w_k = res.xStar["xvars"]

        x_k = w_k[:len(x_k)]
        z_k = w_k[len(x_k):]
        print(f"Full-step g(x,y): {np.sum(np.minimum(sipcc_problem.eval_complementarity_variable(x_k,y_k),0))}")
        print(f"Full-step z bound: {np.sum(np.minimum(sipcc_problem.eval_z_bound(x_k,y_k,z_k),0))}")
        print(f"Full-step complementarity gap: {np.sum(np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k)))}")
        print(f"Full-step inequalities violation: {np.sum(np.minimum(sipcc_problem.eval_inequalities(x_k,y_k,z_k),0))}")
        print(f"Full-step equalities violation: {vectorops.norm_L1(sipcc_problem.eval_equalities(x_k,y_k,z_k))}")
        
        # Line Search
        weight_f = 1.0  #TODO: choose weight so that the SQP direction leads to a descent in the merit function
        converged,w_k,IndexSet_next,total_maximum_violation = \
            lineSearch(sipcc_problem,w_k_init,w_k,dim_x,y_k,f,c,w_lb,w_ub,c_lb,c_ub,weight_f=weight_f,settings=settings)
        x_k = w_k[:len(x_k)]
        z_k = w_k[len(x_k):]
        complementarity_gap = np.sum(np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k)))
        equality_violation = vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k))
        if not converged:
            break
        
        # Delete Index Points if force < threshold and distance > threshold 
        IndexSet_tmp = []
        z_tmp = []
        dist = sipcc_problem.complementarity
        comp = sipcc_problem.get_complementarity_constraint()
        dist.setx(x_k)
        for i in range(0,len(IndexSet)):
            js = dim_x+dim_z*i
            je = dim_x+dim_z*i+dim_z
            index_mag = abs(w_k[js] if sipcc_problem.z_proj is None else sipcc_problem.z_proj.dot(w_k[js:je]))
            distance_mag = np.min(dist.value(x_k,IndexSet[i]))
            #comp_mag = comp(x_k,IndexSet[i],w_k[js:je]) 
            comp_mag = 0
            delete = True
            if index_mag >= settings.index_variable_deletion_magnitude and distance_mag <= settings.index_point_deletion_distance and comp_mag < settings.index_point_deletion_complementarity + complementarity_slack*100:
                delete = False
                IndexSet_tmp.append(IndexSet[i])
                z_tmp.append(w_k[js:je])
            #print("Index",i,"force magnitude",index_mag,"distance mag",distance_mag,"del",delete)
        dist.clearx()    

        # Add penetration points detected in the line search
        for index_point in IndexSet_next:
            if all (sipcc_problem.domain.distance(index_point,idx) > settings.min_index_point_distance for idx in IndexSet_tmp):
                IndexSet_tmp.append(index_point)
                z_tmp.append([0]*dim_z)
        
        print(f"Index Point Deletion (z < {settings.index_variable_deletion_magnitude}, d > {settings.index_point_deletion_distance}):")
        print(f"Points before deletion: {len(y_k)}")

        IndexSet = IndexSet_tmp   
        z_k = list(np.concatenate(z_tmp)) if len(z_tmp) != 0 else []
        y_k = copy.deepcopy(IndexSet)
        
        print(f"Points after deletion: {len(IndexSet)}")  
        
        print(f"Iteration {iters} result:")                   
        print(f"Sum min-distance violation: {total_maximum_violation}")
        print(f"Complementarity gap: {complementarity_gap}")
        print(f"  With dropped...: {np.sum(np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k)))}")
        if sipcc_problem.xyz_eq:
            print(f"Aggregate equality violation: {equality_violation}")
            print(f"  With dropped...: {vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k))}")
        print("=========================================== \n")
    
        iters += 1
    
    if settings.callback:
        settings.callback(x_k,y_k,z_k)

    return x_k,y_k,z_k