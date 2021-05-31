from pyoptsparse import Optimization, OPT, pyOpt_utils

def optimizeMPCC(objective,constraints,x_init,x_min,x_max,constraints_lb,constraints_ub,Major_step_limit,Max_Iters):
    
    dim_var = len(x_init)
    dim_con = len(constraints_lb)
    # Objective function for SNOPT
    def objfunc(xdict):
        x = xdict["xvars"] 
        funcs = {}
        
        funcs["obj"] = objective.value(x)
        
        con = constraints.value(x).reshape((1,-1)).tolist()
        funcs["con"] = con
    
        fail = False
    
        return funcs, fail
    
    # Derivative function for SNOPT
    def sens(xdict, funcs):
        x = xdict["xvars"] 
        
        derivative = pyOpt_utils.convertToCOO(constraints.df_dx(x).todense())
        funcsSens = {"con":{"xvars": derivative},"obj":{"xvars":[0]*len(x)}}
        fail = False
    
        return funcsSens, fail

    # Optimization Object
    optProb = Optimization("MPCC", objfunc)
    
    # Design Variables
    optProb.addVarGroup("xvars", dim_var, "c", lower=x_min, upper=x_max, value=x_init)
 
    # Constraints
    optProb.addConGroup("con", dim_con, lower=constraints_lb, upper=constraints_ub, wrt=["xvars"])
    
    # Objective
    optProb.addObj("obj")
        
    # Optimizer
    optOptions = {'Major step limit':Major_step_limit,"Derivative option":0,"Major iterations limit":Max_Iters,"Minor iterations limit":100,"Scale option":1}
    opt = OPT("SNOPT",options=optOptions)
    
    # Solve
    result = opt(optProb, sens=sens)
    
    return result