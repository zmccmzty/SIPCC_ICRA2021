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

    - d(w) >= 0 the constraint marging w_lb <= w <= w_lb and c_lb <= c(w) <= c_lb,
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
    maximum_violation_before_opt = np.sum(np.abs(np.minimum(maximum_violations,0)))/sipcc_problem.complementarity.scale
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
        # Uncomment this line to do line search only on the configuration variables
        # w_tmp[dim_robo:] = w_k[dim_robo:]
        
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
                
        sipcc_problem.complementarity.setx(x_tmp)
        minvalue_res = sipcc_problem.complementarity.minvalue(x_tmp)
        maximum_violations = [r[0] for r in minvalue_res]
        maximum_violation_pts = [r[1] for r in minvalue_res]
        maximum_violation_tmp = np.sum(np.abs(np.minimum(maximum_violations,0)))/sipcc_problem.complementarity.scale
        sipcc_problem.complementarity.clearx()
        merit_tmp += settings.max_violation_merit_weight*maximum_violation_tmp
            
        if not (merit_tmp < merit_before_opt):
            if alpha==1:
                print(f"Full-step merit value: {merit_tmp}, penetration {maximum_violation_tmp}")
            if maximum_violation_tmp != 0:
                for d,y in zip(maximum_violations,maximum_violation_pts):
                    if d < 0.0 and (y not in IndexSet_next):
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
        maximum_violation_tmp = maximum_violation_before_opt
    if len(IndexSet_next) > 0:
        print(f"  {len(IndexSet_next)} points detected in line search! ")
    
    total_maximum_violation = maximum_violation_tmp
    return converged, w_tmp, IndexSet_next, total_maximum_violation

def optimizeSIPCC(sipcc_problem,x_init,x_lb,x_ub,settings=SIPCCOptimizationSettings()):
    
    res = SIPCCOptimizationResult()
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

    res.x0 = x_init
    res.xlog.append(x_init)
    
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
        if num_added != 0:
            z_k_init = z_k_last + [0]*(dim_z*num_added)
        else:
            z_k_init = z_k_last
        if not settings.warm_start_z:
            z_k_init = [0]*len(z_k_init)
        
        if settings.callback:
            settings.callback(x_k,y_k,z_k_init)
            
        # complementarity_slack = max(min(0.1**(iters),0.001),1e-6)
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
        res_MPCC = optimizeMPCC(f,c,w_k_init,w_lb,w_ub,c_lb,c_ub,Solver,Major_step_limit,min(Max_Iters_init+Max_Iters_growth*iters,50))
        w_k = res_MPCC.xStar["xvars"]
        res.xlog.append(w_k)
        
        x_k = w_k[:len(x_k)]
        z_k = w_k[len(x_k):]
        v1 = np.minimum(sipcc_problem.eval_complementarity_variable(x_k,y_k),0)
        complementarity_constraint_value = np.sum(v1)/sipcc_problem.complementarity.scale
        v2 = np.minimum(sipcc_problem.eval_z_bound(x_k,y_k,z_k),0)
        z_bound = np.sum(v2)
        v3 = np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k))
        complementarity_gap = np.sum(v3)/sipcc_problem.complementarity.scale
        v4 = np.minimum(sipcc_problem.eval_inequalities(x_k,y_k,z_k),0)
        inequality_violation = np.sum(v4)
        v5 = sipcc_problem.eval_equalities(x_k,y_k,z_k)
        equality_violation = vectorops.norm_L1(v5)
        v = np.hstack([v1,v2.reshape(-1),v3,v4,v5])
        print(f"Full-step g(x,y): {complementarity_constraint_value}")
        print(f"Full-step z bound: {z_bound}")
        print(f"Full-step complementarity gap: {complementarity_gap}")
        print(f"Full-step inequalities violation: {inequality_violation}")
        print(f"Full-step equalities violation: {equality_violation}")
        
        sipcc_problem.complementarity.setx(x_k)
        minvalue_res = sipcc_problem.complementarity.minvalue(x_k)
        maximum_violations = [r[0] for r in minvalue_res]
        total_maximum_violation = np.sum(np.abs(np.minimum(maximum_violations,0)))/sipcc_problem.complementarity.scale
        sipcc_problem.complementarity.clearx()
        print(f"Full-step total penetration: {total_maximum_violation}, tolarance: {settings.constraint_violation_tolerance}")
        
        if equality_violation < settings.equality_tolerance \
            and total_maximum_violation < settings.constraint_violation_tolerance\
            and complementarity_gap < settings.complementarity_gap_tolerance:
                converged = True
                break
        
        # Line Search        
        weight_f = 2*np.dot(f.gradient(w_k),(w_k-w_k_init))/vectorops.norm_L2(v)
        converged,w_k,IndexSet_next,total_maximum_violation = \
            lineSearch(sipcc_problem,w_k_init,w_k,dim_x,y_k,f,c,w_lb,w_ub,c_lb,c_ub,weight_f=weight_f,settings=settings)
        print(f"Line search converged {converged}")
        x_k = w_k[:len(x_k)]
        z_k = w_k[len(x_k):]
        complementarity_gap = np.sum(np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k)))/sipcc_problem.complementarity.scale
        equality_violation = vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k))
        
        # if total_maximum_violation > settings.constraint_violation_tolerance:
        #     print("distance constraint scale x2")
        #     sipcc_problem.complementarity.scale *= 2
            
        if not converged:
            iters += 1
            
            # Keep all the points in the index set
            IndexSet_tmp = []
            z_tmp = []
            for i in range(0,len(IndexSet)):
                js = dim_x+dim_z*i
                je = dim_x+dim_z*i+dim_z
                IndexSet_tmp.append(IndexSet[i])
                z_tmp.append(w_k[js:je])
                    
            # Add penetration points detected in the line search
            len_index_before_add = len(IndexSet)
            print(f"number of points before add: {len_index_before_add}")
            print(f"number of points in IndexSet_next: {len(IndexSet_next)}")

            for index_point in IndexSet_next:
                if all (sipcc_problem.domain.distance(index_point,idx) > settings.min_index_point_distance for idx in IndexSet_tmp):
                    IndexSet_tmp.append(index_point)
                    z_tmp.append([0]*dim_z)
            
            IndexSet = IndexSet_tmp   
            z_k = list(np.concatenate(z_tmp)) if len(z_tmp) != 0 else []
            y_k = copy.deepcopy(IndexSet)
            len_index_after_add = len(IndexSet)
            print(f"number of points after add: {len_index_after_add}")
            if len_index_before_add == len_index_after_add:
                converged = False
                break
            else:
                continue

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
            distance_mag = dist.value(x_k,IndexSet[i])/sipcc_problem.complementarity.scale
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
        res.IndexSum += len(y_k)
        
        IndexSet = IndexSet_tmp   
        z_k = list(np.concatenate(z_tmp)) if len(z_tmp) != 0 else []
        y_k = copy.deepcopy(IndexSet)
        
        print(f"Points after deletion: {len(IndexSet)}")  
        res.activeIndexSum += len(IndexSet)
        
        print(f"Iteration {iters} result:")                   
        print(f"Sum min-distance violation: {total_maximum_violation}")
        print(f"Complementarity gap: {complementarity_gap}")
        print(f"  With dropped...: {np.sum(np.abs(sipcc_problem.eval_complementarity_gap(x_k,y_k,z_k)))/sipcc_problem.complementarity.scale}")
        if sipcc_problem.xyz_eq:
            print(f"Aggregate equality violation: {equality_violation}")
            print(f"  With dropped...: {vectorops.norm_L1(sipcc_problem.xyz_eq.value(x_k,y_k,z_k))}")
        print("=========================================== \n")
    
        iters += 1
    
    if settings.callback:
        settings.callback(x_k,y_k,z_k)

    res.num_iterations = iters
    if iters < settings.max_iters and converged == True:
        res.status = "converged"
        print("Success")
    else:
        print("Fail")
    print(f"total_maximum_violation: {total_maximum_violation}")
    print(f"Aggregate equality violation: {equality_violation}")
    print(f"Complementarity gap: {complementarity_gap}")
        
    res.total_maximum_violation = total_maximum_violation
    res.InfiniteAggregateConstraint_violation = equality_violation
    res.complementarity_gap = complementarity_gap    
    res.result['x_star'] = x_k
    res.result['y_star'] = y_k
    res.result['z_star'] = z_k

    return res