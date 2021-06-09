from pyoptsparse import Optimization, OPT, pyOpt_utils
import scipy.sparse
import numpy as np

def optimizeMPCC(objective,constraints,x_init,x_min,x_max,constraints_lb,constraints_ub,Solver,Major_step_limit,Max_Iters):
    
    dim_var = len(x_init)
    dim_con = len(constraints_lb)
    # Objective function for Optimizer
    def objfunc(xdict):
        x = xdict["xvars"] 
        funcs = {}
        
        funcs["obj"] = objective.value(x)
        
        constraints.setx(x)
        con = constraints.value(x)
        assert isinstance(con,np.ndarray) and len(con.shape)==1,"Constraints must return a vector"
        constraints.clearx()
        funcs["con"] = con
    
        fail = False
    
        return funcs, fail
    
    df0 = constraints.df_dx(x_init)
    if scipy.sparse.issparse(df0):
        mat = df0.tocoo()
        df0 = {"coo": [mat.row, mat.col, mat.data], "shape": mat.shape}
        #print("Sparsity",mat.nnz,"/",np.product(mat.shape),"=",mat.shape,"entries")
    else:
        df0 = pyOpt_utils.convertToCOO(df0)
    rows = df0['coo'][0]
    cols = df0['coo'][0]
    
    # Derivative function for Optimizer
    def sens(xdict, funcs):
        x = xdict["xvars"] 
        
        constraints.setx(x)
        df = constraints.df_dx(x)
        constraints.clearx()
        if scipy.sparse.issparse(df):
            if df.nnz != len(rows):
                assert df.nnz == len(rows),"Invalid sparsity pattern in jacobian? {} != {}".format(df.nnz,len(rows))
            mat = df.tocoo()
            assert len(mat.data) == len(rows),"Invalid sparsity pattern in jacobian? {} != {}".format(len(mat.data),len(rows))
            data = mat.data
        else:
            data = df.flatten()
            assert len(data) == len(rows),"Invalid sparsity pattern in jacobian? {} != {}".format(len(data),len(rows))
        derivative = {'coo':[rows,cols,data],'shape':df0['shape']}
        objgrad = objective.gradient(x)
        funcsSens = {"con":{"xvars": derivative},"obj":{"xvars":objgrad}}
        fail = False
    
        return funcsSens, fail

    # Optimization Object
    optProb = Optimization("MPCC", objfunc)
    
    # Design Variables
    optProb.addVarGroup("xvars", dim_var, "c", lower=x_min, upper=x_max, value=x_init)
 
    # Constraints
    optProb.addConGroup("con", dim_con, lower=constraints_lb, upper=constraints_ub, wrt=["xvars"], jac={'xvars':df0})
    
    # Objective
    optProb.addObj("obj")
        
    # Optimizer
    if Solver=='SNOPT':
        optOptions = {'Major step limit':Major_step_limit,"Derivative option":0,"Major iterations limit":Max_Iters,"Minor iterations limit":100,"Scale option":1}
        opt = OPT("SNOPT",options=optOptions)
    else:
        optOptions = {'max_iter':Max_Iters}
        opt = OPT("IPOPT",options=optOptions)
    
    # Solve
    result = opt(optProb, sens=sens)
    
    return result