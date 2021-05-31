import time
import math
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.io
import heapq
import random
from klampt.math import vectorops
# =============================================================================
# Install OSQP
# pip install osqp
# conda install -c conda-forge osqp
# https://osqp.org/docs/get_started/index.html
# =============================================================================
try:
    import osqp
    OSQP_ENABLED = True
except ImportError:
    OSQP_ENABLED = False
    raise NotImplementedError("OSQP is currently required.  TODO: implement other solvers.")

DEBUG_GRADIENTS = False
DEBUG_TRAJECTORY_INITIALIZATION = False

# Whether to use a line search or trust region to control the step size
STEP_SIZE_METHOD = 'line search'
#STEP_SIZE_METHOD = 'trust region'

# Whether new constraints are detected during line search / trust region validation
DETECT_NEW_COLLISIONS_IN_STEP = True

# How are new constraints generated
#ORACLE_METHOD = 'most violating'
ORACLE_METHOD = 'most violating all'
#ORACLE_METHOD = 'random 30'
#ORACLE_METHOD = 'all'

# Whether drop constraints whose value is bigger than a given threshold
REMOVE_CONSTRAINTS = True

# How is the QP problem solved
#QP_SOLVE_METHOD = 'custom'
QP_SOLVE_METHOD = 'osqp'

# How to measure the score of the constraints' value
#SCORING_METRIC = 'L2'
#SCORING_METRIC = 'L2-average'
#SCORING_METRIC = 'L1'
#SCORING_METRIC = 'L1-average'
SCORING_METRIC = 'minimum'
INCLUDED_CONSTRAINTS = 'all'
INCLUDED_CONSTRAINT_PARAMETERS = 'all'          #include all possible constraint parameters
#INCLUDED_CONSTRAINT_PARAMETERS = 'deepest'       #testing: only include deepest constraint parameter

class SemiInfiniteOptimizationSettings:
    """Settings for the semi-infinite programming solver.

    Attributes:
    - max_iters: the maximum number of outer iterations (default 50)
    - xepsilon: terminate with success if the x variable does not change more than this amount
      between iterations (default 1e-4)
    - constraint_inflation: the constraints will be inflated by this amount to help get a feasible solution
    - constraint_drop_value: drop constraints from the index set when their values exceed this value
    - minimum_constraint_value: add a hard constraint to prevent the optimizer from reaching any state in
      which the constraint function is below this value (useful for preventing local minima in deep penetrations)
    - initial_objective_score_weight: initial weight of the objective function in the merit function
    - minimum_objective_score_weight: lower bound on the weight of the objective function in the merit function
    - parameter_exclusion_distance: a new index parameter will not be added if it is within this distance of a
      previously instantiated parameter.
    - qp_solver_params: dictionary of any parameters used in the QP solver.  Right now, only 'regularizationFactor'
      is used.
    """
    def __init__(self):
        self.max_iters = 50
        self.xepsilon = 1e-4
        self.constraint_inflation = 1e-3
        self.constraint_drop_value = 0.1 
        #self.constraint_drop_value = float('inf')  #set this to infinity to never drop constraints
        self.minimum_constraint_value = -float('inf')
        self.initial_objective_score_weight = 1e-1
        self.initial_penalty_parameter = 1e-1
        self.minimum_objective_score_weight = 1e-5
        self.parameter_exclusion_distance = 1e-3
        self.qp_solver_params = {'regularizationFactor':1e-3}

def numeric_gradient(f,x,h=1e-4,method='centered'):
    """h is the step size"""
    xtest = np.array(x,dtype=float)
    g = np.zeros(len(x))
    if method == 'centered':
        for i in range(len(x)):
            xtest[i] += h
            b = f(xtest)
            xtest[i] -= 2*h
            a = f(xtest)
            xtest[i] = x[i]

            g[i] = (b-a)
        g *= 1.0/(2*h)
        return g
    elif method == 'forward':
        a = f(x)
        for i in range(len(x)):
            xtest[i] += h
            b = f(xtest)

            g[i] = (b-a)
        g *= 1.0/h
        return g
    elif method == 'backward':
        a = f(x)
        for i in range(len(x)):
            xtest[i] -= h
            b = f(xtest)

            g[i] = (a-b)
        g *= 1.0/h
        return g
    else:
        raise ValueError("Invalid method %s"%(method,))

def test_gradient(f,fgrad,x,h=1e-4,epsilon=1e-3,name=""):
    """Prints whether the gradient of f is calculated correctly via fgrad at x.
    Errors are printed out if the approximation error exceeds epsilon.
    """
    g_numeric = numeric_gradient(f,x,h)
    g_calc = fgrad(x)
    diff = g_numeric-g_calc
    valid = True
    for i,v in enumerate(diff):
        if abs(v) > epsilon:
            print(("test_gradient %s: errorneous value %d: numeric %g vs function %g"%(name,i,g_numeric[i],g_calc[i])))
            valid = False
    return valid

def numeric_jacobian(f,x,h=1e-4,method='centered'):
    """h is the step size"""
    assert h > 0
    xtest = np.array(x,dtype=float)
    fx0 = f(x)
    d = len(fx0)
    g = np.zeros((d,len(x)))
    if method == 'centered':
        for i in range(len(x)):
            xtest[i] += h
            b = f(xtest)
            xtest[i] -= 2*h
            a = f(xtest)
            xtest[i] = x[i]
            g[:,i] = (b-a)
        g *= 1.0/(2*h)
        return g
    elif method == 'forward':
        for i in range(len(x)):
            xtest[i] += h
            b = f(xtest)

            g[:,i] = (b-fx0)
        g *= 1.0/h
        return g
    elif method == 'backward':
        for i in range(len(x)):
            xtest[i] -= h
            b = f(xtest)

            g[:,i] = (a-fx0)
        g *= 1.0/h
        return g
    else:
        raise ValueError("Invalid method %s"%(method,))

def test_jacobian(f,fgrad,x,h=1e-4,epsilon=1e-3,name=""):
    """Prints whether the Jacobian of f is calculated correctly via fgrad at x.
    Errors are printed out if the approximation error exceeds epsilon.
    """
    g_numeric = numeric_jacobian(f,x,h)
    g_calc = fgrad(x)
    diff = g_numeric-g_calc
    valid = True
    for i in range(diff.shape[1]):
        if np.any(np.abs(diff[:i]) > epsilon):
            print(("test_gradient {}: errorneous value {}: numeric {} vs function {}",format(name,i,g_numeric[:,i],g_calc[:,i])))
            valid = False
    return valid
        
class ObjectiveFunctionInterface:
    """Base class for an objective function."""
    def __init__(self):
        pass
    def value(self,x):
        """Returns the objective function value at x"""
        raise NotImplementedError()
    def minstep(self,x):
        """Returns a shift dx that locally minimizes f(x+dx) up to second order"""
        raise NotImplementedError()
    def hessian(self,x):
        """Compute / approximate the hessian of objective at x.  Letting H=hessian(),
        the QP optimizer will try to minimize ||*(dx-minstep()||_H^2 = (dx-minstep())^T H (dx-minstep()).

        A scalar return value is OK too, and is interpreted as a scaling of the identity matrix.

        A vector return value is OK too, and is interpreted as W=diag(hessian()).
        """
        return 1
    def integrate(self,x,dx):
        return np.asarray(x)+np.asarray(dx)

class ConstraintInterface:
    """A constraint is a function of the form f(x) >= 0 where x is in R^n.

    To allow for caching between multiple constraints, to retrieve f(x) the methods setx,
    value, and clearx are called in the format:
      f.setx(x)
      fx = f.value(x)
      f.clearx()
      #result is in fx
    """
    def __init__(self):
        pass
    def dims(self):
        """Returns the number of dimensions of f(x).  0 indicates a scalar."""
        return 0
    def __call__(self,x):
        """Evaluates f(x)"""
        self.setx(x)
        fx = self.value(x)
        self.clearx()
        return fx
    def setx(self,x):
        """Called with the x value before value or df_dx Can set a cache here."""
        pass
    def clearx(self):
        """Called after all calls to value or df_dx with a given x.  Can clear a cache here."""
        pass
    def value(self,x):
        """Returns the function value at the optimization variable x. x must be previously set using setx."""
        raise NotImplementedError()
    def df_dx(self,x):
        """Returns the Jacobian or gradient of the value with respect to x.
        x must be previously set using setx."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearx()
        res = numeric_gradient(lambda v:self.__call__(v),x,h)
        self.setx(x)
        return res

class DomainInterface:
    """A representation of some domain for use in SemiInfiniteConstraint.domain()"""
    def sample(self):
        raise NotImplementedError()
    def extrema(self):
        raise NotImplementedError()

class IntervalDomain(DomainInterface):
    """A 1D interval"""
    def __init__(self,vmin,vmax):
        self.interval = (vmin,vmax)
    def sample(self):
        return random.uniform(self.vmin,self.vmax)
    def extrema(self):
        return [self.vmin,self.vmax]

class SetDomain(DomainInterface):
    """A discrete set of items"""
    def __init__(self,items):
        self.items = list(items)
    def sample(self):
        return random.choice(self.items)
    def extrema(self):
        return self.items

class SemiInfiniteConstraintInterface:
    """A semi-infinite constraint is of the form f(x,y) >= 0 for all y in Y, where
    x is in R^n and Y is a subset of R^m.

    To allow for caching between multiple constraints, to retrieve f(x,y) the methods setx,
    value, and clearx are called in the format:
      f.setx(x)
      fxy = f.value(x,y)
      f.clearx()
      #result is in fxy
    """
    def __init__(self):
        pass
    def dims(self):
        """Returns the number of dimensions of f(x).  0 indicates a scalar."""
        return 0
    def __call__(self,x,y):
        """Evaluates f(x,y).  No need to override this"""
        self.setx(x)
        res = self.value(x,y)
        self.clearx()
        return res
    def eval_minimum(self,x):
        """Helper: evaluates min_{y in Y} f(x,y) without caching."""
        self.setx(x)
        newpts = self.minvalue(x)
        self.clearx()
        if len(newpts) > 0 and not hasattr(newpts[0],'__iter__'): #it's just a plain pair
            newpts = [newpts]
        for (dmin,param) in newpts:
            return dmin
        raise ValueError("minvalue does not appear to return a minimum value...")

    def setx(self,x):
        """Called with the x value before value, minvalue, oracle, df_dx, or df_dy.
        Can set a cache here."""
        pass
    def clearx(self):
        """Called after all calls to value, minvalue, oracle, df_dx, or df_dy with
        a given x.  Can clear a cache here."""
        pass
    def value(self,x,y):
        """Returns the function value at the optimization variable x and parameter y.
        x must be previously set using setx."""
        raise NotImplementedError()
    def minvalue(self,x,bound=None):
        """Returns a pair (fmin,param) containing (min_y f(x,y), arg min_y f(x,y)).
        In other words, fmin=f(x,param).

        The optimizer will add these values to the constraint list.

        x must be previously set using setx.

        Alternatively, can return a list of such pairs.  This may be useful when you can
        efficiently produce multiple local minima."""
        raise NotImplementedError()
    def df_dx(self,x,y):
        """Returns the Jacobian or gradient of the value with respect to x.
        x must be previously set using setx."""
        raise NotImplementedError()
    def df_dy(self,x,y):
        """Returns the Jacobian of the value with respect to param. Not currently used"""
        raise NotImplementedError()
    def domain(self):
        """Returns some representation of the domain.  See sampleDomain and domainExtrema
        allowable values."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,y,h=1e-4):
        """Helper function: properly evaluates df_dx using numeric differentiation"""
        self.clearx()
        res = numeric_gradient(lambda v:self.__call__(v,y),x,h)
        self.setx(x)
        return res
    def df_dy_numeric(self,x,y,h=1e-4):
        """Helper function: properly evaluates df_dy using numeric differentiation"""
        return numeric_gradient(lambda v:self.value(x,v),y,h)

class MinimumConstraintAdaptor(ConstraintInterface):
    """Given a semi-infinite constraint, returns the standard constraint f(x) = min_y f(x,y) >= 0. """
    def __init__(self,semi_infinite_constraint):
        self.f = semi_infinite_constraint
        self.minval = None
        self.minparam = None
    def dims(self):
        return self.f.dims()
    def setx(self,x):
        self.f.setx()
        res = self.minvalue(x)
        if len(res) > 0 and hasattr(res[0],'__iter__'):
            res = res[0]
        self.minval,self.minparam = res
    def clearx(self):
        self.f.clearx()
    def value(self,x):
        return self.minval
    def df_dx(self,x):
        return self.f.df_dy(x,self.minparam)

class SemiInfiniteConstraintAdaptor(SemiInfiniteConstraintInterface):
    """Turns a standard subclass of ConstraintInterface into a SemiInfiniteConstraintInterface
    with domain {0}
    """
    def __init__(self,constraint):
        self.constraint = constraint
    def dims(self):
        return self.constraint.dims()
    def setx(self,x):
        self.constraint.setx(x)
    def clearx(self):
        self.constraint.clearx()
    def value(self,x,y):
        return self.constraint.value(x)
    def minvalue(self,x,bound=None):
        return (self.constraint.value(x),0)
    def df_dx(self,x,y):
        return self.constraint.df_dx(x)
    def df_dy(self,x,y):
        d = self.dims()
        if d == 0: return 0
        return np.zeros(d)
    def domain(self):
        return SetDomain([0])
    

class SemiInfiniteOptimizationResult:
    """Stores the results from a SIP problem

    Members:
    - x0: the initial state
    - fx0: the initial objective function value
    - gx0: the initial constraint value
    - x: the solved state
    - fx: the solved objective function value
    - gx: the solved constraint value
    - instantiated_params: the list of instantiated semi-infinite parameters
    - trace: the sequence of states
    - num_iterations: the number of iterations
    - status: 'optimum', 'local optimum', 'not converged', or 'error'
    - time: the total time for optimization
    - time_cons: the time for constraint queries
    """
    def __init__(self):
        self.x0 = None
        self.fx0 = None
        self.gx0 = None
        self.x = None
        self.fx = None
        self.gx = None
        self.instantiated_params = []
        self.trace = []
        self.num_iterations = 0
        self.status = 'not converged'
        self.time = 0
        self.time_cons = 0


def scoring_metric(v):
    if SCORING_METRIC == 'L1':
        return sum(abs(x) for x in v)
    elif SCORING_METRIC == 'L1-average':
        return sum(abs(x) for x in v)/len(v)
    elif SCORING_METRIC == 'L2':
        return np.dot(v,v)
    elif SCORING_METRIC == 'L2-average':
        return np.dot(v)/len(v)
    else:
        r = min(v)
        if r >= 0: return 0
        return abs(r)

def scoring_metric2(v):
    """Ignores positive values"""
    v = [d for d in v if d < 0]
    if len(v) == 0: return 0
    return scoring_metric(v)

def score(objective,constraints,x,constraint_generation_data,objScoreWeight,discoverNew=False):
    """Returns a score for the line search that weights the objective and the constraint values.
    If discoverNew = True, then this tries identifying a new constraint parameter.
    Return value: (score,fx,gx)
    """
    for c in constraints:
        c.setx(x)
    cdata = constraint_generation_data
    fx = objective.value(x)
    gx = np.zeros(len(constraints))
    gscore = 0.0
    for i,c in enumerate(constraints):
        if len(cdata.instantiated_params[i]) == 0:
            gx[i] = float('inf')
            gvals = []
        else:
            gvals = [c.value(x,y) for y in cdata.instantiated_params[i]]
            gx[i] = min(gvals)
            gvals = [g for g in gvals if g < 0]
        if discoverNew:
            newpts = c.minvalue(x,gx[i])
            if len(newpts) > 0 and not hasattr(newpts[0],'__iter__'): #it's just a plain pair
                newpts = [newpts]
            for (dmin,newparam) in newpts:
                if dmin < gx[i]:
                    gx[i] = dmin
                    if cdata.instantiate(i,newparam):
                        #print("INSIDE SCORE: ADDING NEW CONSTRAINT AT VALUE",dmin)
                        if dmin < 0:
                            gvals.append(dmin)
        if len(gvals) > 0:
            gscore += scoring_metric(gvals)
    for c in constraints:
        c.clearx()
    score = gscore + objScoreWeight*fx
    #score = (gscore,fx)
    return (score,fx,gx)
    
def _score(objectiveVal,constraintVals,objScoreWeight):
    """Evaluates the score for a given objective function value and constraint values"""
    #return vectorops.normSquared([d for d in constraintVals if d < 0]) + objScoreWeight*objectiveVal
    gscore = 0
    for cs in constraintVals:
        gscore += scoring_metric2(cs)
    #return (gscore,objectiveVal)
    return gscore + objScoreWeight*objectiveVal

class ConstraintGenerationData:
    """Stores intermediate data during SIP"""
    def __init__(self,constraints):
        self.constraints = constraints
        self.instantiated_params = [[] for c in constraints]
        self.visited_points = [set() for c in constraints]
        self.exclusion_distance = 1e-3
        self.constraint_inflation = 1e-5
        self.constraint_drop_value = float('inf')

    def oracle(self,x,update_x=True,method='most violating all'):
        """Returns the list of lists of depths for each instantiated parameter"""
        if not update_x:
            for c in self.constraints:
                c.setx(x)
        depths = []
        dlist = []
        for i,c in enumerate(self.constraints):
            cdepths = [c.value(x,y) for y in self.instantiated_params[i]]
            if method is not None and REMOVE_CONSTRAINTS:
                cdepths_drop = [v for v in cdepths if v < self.constraint_drop_value]
                if len(cdepths_drop) < len(cdepths):
                    print(("Dropping",len(cdepths)-len(cdepths_drop),"instantiated parameters of constraint",i))
                    print((cdepths,"<",self.constraint_drop_value,"gives",cdepths_drop))
                    self.instantiated_params[i] = [y for v,y in zip(cdepths,self.instantiated_params[i]) if v < self.constraint_drop_value]
                    cdepths = cdepths_drop
                    #revise visited_poitns
                    self.visited_points[i] = set()
                    for y in self.instantiated_params[i]:
                        yindex = tuple(int(v/self.exclusion_distance) for v in y)
                        self.visited_points[i].add(yindex)

            olddmin = (0 if len(cdepths)==0 else min(0,min(cdepths)))
            if method is None:
                pass
            elif method.startswith('most violating'):
                newpts = c.minvalue(x,olddmin)
                if len(newpts) > 0 and not hasattr(newpts[0],'__iter__'): #it's just a plain pair
                    newpts = [newpts]

                #testing
                """
                for (dmin,newparam) in newpts:
                    dtest = c.value(x,newparam)
                    if abs(dtest-dmin) > 1e-4:
                        print "Strange, the %s.value function differs from %s.minvalue: %g vs %g"%(c.__class__.__name__,c.__class__.__name__,dtest,dmin)
                        raw_input("Press enter to continue > ")
                """

                if method.endswith('all'):
                    for (dmin,newparam) in newpts:
                        if dmin < olddmin:
                            if self.instantiate(i,newparam):
                                cdepths.append(dmin)
                else:
                    for (dmin,newparam) in newpts:
                        dlist.append((dmin,i,newparam))
            elif method == 'all':
                if len(self.instantiated_params[i]) == 0:
                    Y = c.domain()
                    if isinstance(Y,(list,tuple)):
                        elements = [domainExtrema(Yi) for Yi in Y]
                        newparam = [(a, b) for a in elements for b in elements]
                    else:
                        newparam = domainExtrema(Y)
                    newparam = [random.choice(newparam) for k in range(1000)]
                    print(("ORACLE-ALL: instantiating",len(newparam),"parameters"))
                    for y in newparam:
                        if self.instantiate(i,y):
                            cdepths.append(c.value(i,y))
            else:
                assert method.startswith('random'),"Can only do most violating, most violating all, or random constraint generation at the moment"
                Y = c.domain()
                nsamples = 1
                if method != 'random':
                    nsamples = int(method[7:])
                for k in range(nsamples):
                    if isinstance(Y,(list,tuple)):
                        elements = [sampleDomain(Yi) for Yi in Y]
                        newparam = np.flatten(elements)
                    else:
                        newparam = sampleDomain(Y)
                    if self.instantiate(i,newparam):
                        cdepths.append(c.value(i,newparam))
            depths.append(cdepths)
        if method == 'most violating':
            if len(dlist) > 0:
                (dmin,i,newparam) = min(dlist)
                if self.instantiate(i,newparam):
                    depths[i].append(dmin)
        if not update_x:
            for c in self.constraints:
                c.clearx()
        return depths

    def instantiate(self,cindex,y):
        """Instantiates a new parameter y for constraint cindex.  Returns True if it is a novel parameter according to exclusion_distance."""
        yindex = tuple(int(v/self.exclusion_distance) for v in y)
        if yindex not in self.visited_points[cindex]:
            self.instantiated_params[cindex].append(y)
            self.visited_points[cindex].add(yindex)
            return True
        return False

    def solve_qp(self,W,xdes,A,b,xmin=None,xmax=None,**kwargs):
        if len(A) == 0: return xdes
        if QP_SOLVE_METHOD == 'osqp':
            return self.solve_qp_osqp(W,xdes,A,b,xmin,xmax,**kwargs)
        else:
            return self.solve_qp_custom(W,xdes,A,b,xmin,xmax,**kwargs)

    def solve_qp_osqp(self,W,xdes,A,b,xmin=None,xmax=None,verbose=0,regularizationFactor=1e-2,slack_penalty='auto'):
        """Solves for min ||x-xdes||_W^2 + regularizationFactor ||x||^2 s.t., Ax + b >= 0, and optionally xmin <= x <= xmax.
        If the problem is infeasible, it solve a different problem in which slack variables s are added and slack_penalty*||s||^2
        are included.
        """
        #||x-xdes)||_W^2 + reg||x||^2 = (x-xdes)^T W (x-xdes) + x^T (reg I) x = 
        #x^T (W + reg) x - 2 x^T W xdes + const
        #equivalent to solving 1/2 x^T P x + q^T x   with P = W+reg and q = -W*xdes
        solver = osqp.OSQP()
        n = len(xdes)
        m = len(b)
        if isinstance(W,(int,float)):
            P = scipy.sparse.diags([W]*n,format='csc')
        elif isinstance(W,list) or len(W.shape)==1:
            P = scipy.sparse.diags(W,format='csc')
        else:
            assert len(W.shape)==2
            P = W
        q = -P.dot(xdes)
        if regularizationFactor != 0:
            P = P + scipy.sparse.diags([regularizationFactor]*n,format='csc')
        #A = scipy.sparse.csc_matrix(np.array(A))
        #this fails sometimes?
        #A = scipy.sparse.vstack(A,format='csc')
        A = scipy.sparse.vstack([scipy.sparse.coo_matrix(v) for v in A],format='csc')
        b = np.asarray(b)
        l = -b + np.ones(m)*self.constraint_inflation
        u = np.array([np.inf]*m)
        if xmin is not None:
            assert xmax is not None
            A = scipy.sparse.vstack((A,scipy.sparse.diags([1]*n,format='csc')),format='csc')
            l = np.hstack((l,xmin))
            u = np.hstack((u,xmax))
        solver.setup(P=P,q=q,A=A,l=l,u=u,verbose=False,eps_abs=1e-5)
        bnorm = 1
        if any(v < 0 for v in b):
            bnorm = np.linalg.norm([v for v in b if v < 0])
        results = solver.solve()
        if results.x[0] == None or np.linalg.norm(results.x)*regularizationFactor > bnorm:
        #if True:
            if slack_penalty == 'auto':
                bmin = np.min(b)
                slack_penalty = (1+len([v for v in b if v < 0]))/(abs(bmin) + 1e-3)
            """
            if results.x[0] == None:
                print "STRICT LP PROBLEM IS INFEASIBLE, ADDING SLACK PENALTY",slack_penalty
            elif np.linalg.norm(results.x)*regularizationFactor > bnorm:
                print "STRICT LP PROBLEM GIVES LARGE RESULT, ADDING SLACK PENALTY",slack_penalty
            """
            #infeasible, try solving a different problem with slack variables
            #print "STRICT LP PROBLEM IS INFEASIBLE, ADDING SLACK PENALTY",slack_penalty
            P = scipy.sparse.bmat([[P,None],[None,scipy.sparse.diags([slack_penalty]*m,format='csc')]],format='csc')
            q = np.hstack((q,np.zeros(m)))
            Aslack = scipy.sparse.diags([1]*m,format='csc')
            if xmin is not None:
                Aslack = scipy.sparse.vstack((Aslack,scipy.sparse.csc_matrix((n,m))),format='csc')
            A = scipy.sparse.hstack((A,Aslack),format='csc')
            #print "P:",P
            #print "A:",A
            #print "q:",q
            #print "l:",l
            solver = osqp.OSQP()
            solver.setup(P=P,q=q,A=A,l=l,u=u,verbose=False,eps_abs=1e-5)
            results = solver.solve()
            if results.x[0] == None:
                print("ERROR SOLVING QP PROBLEM")
                print(("  P:",P))
                print(("  q:",q))
                print(("  A:",A))
                print(("  l:",l))
                print(("  u:",u))
            #print "Slack variables",results.x[len(xdes):]
            #print "  result",results.x[:len(xdes)]
        else:
            residual = A.dot(results.x)[:m] + b
            """
            #TEST FOR QP SOLVE ISSUES
            if min(residual) < -1e-5:
                print "   Result.pri_res",results.info.pri_res
                print "   Result.dua_res",results.info.dua_res
                print "   Predicted constraint violations after QP solve:",residual
                raw_input()
            """
        x = results.x
        return results.x[:len(xdes)]

    def solve_qp_custom(self,W,xdes,A,b,xmin=None,xmax=None,verbose=0,regularizationFactor=1e-2):
        """Solves for min ||x-xdes||_W^2 + regularizationFactor ||x||^2 s.t., Ax + b >= 0, and optionally xmin <= x <= xmax
        """
        #Given some active constraints, want to solve for min ||x-xdes)||_W^2 + reg ||x||^2 s.t. Aact * x + bact = 0
        #The solution satisfies
        #  2 W(x-xdes) + 2reg x + Aact^T lambda = 0
        #  Aact * x + bact = 0 
        #for some lambda
        # condition Aact by the cholesky decomposition LL^T = (W + regularizationFactor), and let A' = Aact L^-T
        n = len(xdes)
        A = np.asarray(A)
        b = np.asarray(b)
        dpred = np.dot(A,xdes) + b
        inside = []
        for i in range(len(b)):
            if dpred[i] <= 0 or b[i] < 0:
                inside.append(i)
        if xmin is not None:
            assert xmax is not None,"Both xmin and xmax must be specified, or neither"
            k = len(b)
            Afull = [A]
            bfull = [b]
            for i in range(n):
                Afull.append([0.0]*n)
                Afull[-1][i] = 1
                bfull.append(-xmin[i])
                if xmin[i] == xmax[i]:
                    inside.append(k)
                    k += 1
                    break
                if 0 < xmin[i] or xdes[i] <= xmin[i]:
                    inside.append(k)
                k += 1
                Afull.append([0.0]*n)
                Afull[-1][i] = -1
                bfull.append(xmax[i])
                if 0 > xmax[i] or xdes[i] >= xmax[i]:
                    inside.append(k)
                k += 1
            Afull = np.vstack(Afull)
            bfull = np.hstack(bfull)
            dpred = np.dot(Afull,xdes) + bfull
        else:
            Afull,bfull = A,b

        if len(inside) == 0:
            if verbose>=2: print("No constraints assumed to be active?")
            return xdes
        #solve one step of a Quadratic Program centered around W,V
        arows = Afull[inside]
        adepths = bfull[inside]
        aoffsets = dpred[inside]
        if len(inside) > n:
            depthorder = sorted((d,i) for (i,d) in enumerate(aoffsets))[:n]
            depthorder = [i for (d,i) in depthorder]
            arows = arows[depthorder]
            adepths = adepths[depthorder]
            aoffsets = aoffsets[depthorder]
        #expansion term
        aoffsets = aoffsets - np.ones(len(aoffsets))*self.constraint_inflation
        if verbose>=2: print(("  Active constraint depths:",adepths))
        #with weighting, we have 
        #0 = W(A deltax + d0 + A dxdes) => deltax = - (WA)^+ W(d0 + A dxdes)
        Aactive = np.vstack(arows)
        bactive = np.hstack(aoffsets)
        #generalized least squares solution
        scaling = None
        if not isinstance(W,(int,float)):
            if isinstance(W,list) or len(W.shape)==1:
                assert len(W) == Aactive.shape[1]
                #scale by 1/(wi + regularizationFactor)
                scaling = np.zeros(len(W))
                for i in range(len(W)):
                    scaling[i] = 1.0/math.sqrt(W[i] + regularizationFactor)
                    Aactive[:,i] *= scaling[i]
            else:
                #a matrix
                H = W[:,:]
                for i in range(H.shape[0]):
                    H[i,i] += regularizationFactor
                L = np.linalg.cholesky(H)
                Linv = scipy.linalg.solve_triangular(L,np.eye(H.shape[0]),lower=True)
                scaling = Linv
                Aactive = np.dot(Aactive,scaling)
        #lsqrres = scipy.sparse.linalg.lsqr(A,b,damp=math.sqrt(objScoreWeight),iter_lim=10)
        lsqrres = scipy.sparse.linalg.lsqr(scipy.sparse.csr_matrix(Aactive),bactive,iter_lim=20,damp=regularizationFactor)
        res = lsqrres[0]
        if scaling is not None:
            if len(scaling.shape) == 1:
                #diagonal
                res = np.multiply(scaling,res)
            else:
                #matrix
                res = np.dot(scaling,res)
        x = xdes-res
        if xmin is not None:
            #clamp
            x = np.minimum(xmax,np.maximum(xmin,x))
        #always close to 0
        #print "Custom solve predicted residuals",A.dot(x)+b
        return x


def optimizeStandard(objective,constraints,xinit,xmin=None,xmax=None,settings=None,verbose=1):
    """Optimizes a standard NLP using the same framework as the SIP solver"""
    res = SemiInfiniteOptimizationResult()
    res.x0 = xinit
    
    print(xinit)
    x = xinit
    res.trace = [xinit]
    if settings is None:
        settings = SemiInfiniteOptimizationSettings()
    objScoreWeight = settings.initial_objective_score_weight
    penalty_parameter = settings.initial_penalty_parameter
    res.fx0 = objective.value(x)
    res.gx0 = np.zeros(len(constraints))
    res.x = x
    res.fx = res.fx0
    res.gx = np.zeros(len(constraints))
    res.instantiated_params = [[]]
    xepsilon2 = settings.xepsilon**2*len(x)
    xepsilon = math.sqrt(xepsilon2)
    cdata = ConstraintGenerationData(constraints)
    cdata.constraint_inflation = settings.constraint_inflation
    minimum_constraint_value = settings.minimum_constraint_value
    score_orig_trace = []
    score_after_trace = []
    gx_trace = []
    trust_region_size = 0.5

    tstart = time.time()
    tgeom = 0
    iters = 0
    for iters in range(settings.max_iters):
        dxdes  = np.asarray(objective.minstep(x))
        #discover constraint points
        depths = []
        rows = []
        res.gx = np.zeros(len(constraints))
        for i,c in enumerate(constraints):
            c.setx(x)
            res.gx[i] = c.value(x)
            c.value()
            if iters == 0:
                res.gx0[i] = res.gx[i]
                if res.gx0[i] < settings.minimum_constraint_value:
                    print(("WARNING: initial constraint value %d is below the minimum %g < %g"%(i,res.gx0[i],settings.minimum_constraint_value)))
                    minimum_constraint_value = res.gx0[i]
                    if verbose >= 1:
                        eval(input("Press enter to continue > "))
            df_dT = c.df_dx(x)
            c.clearx()
            depths.append(res.gx[i])
            rows.append(df_dT)
        if iters == 0:
            if verbose>=1: print(("Beginning constrained optimization at f(x) =",res.fx0,"g(x) =",res.gx0))
              
        #TEST GRADIENTS
        if DEBUG_GRADIENTS and iters == 1:
            if verbose >= 2:
                for i,c in enumerate(constraints):
                    c.setx(x)
                    f0 = c.value(x)
                    retval = c.df_dx(x)
                    c.clearx()
                    print(("CONSTRAINT GRADIENT MAGNITUDE",np.linalg.norm(retval)))
                    print(("t = 0 :",f0))
                    for j in range(1,11):
                        xnext = objective.integrate(x,np.asarray(retval)*0.01*j)
                        c.setx(xnext)
                        f = c.value(x)
                        c.clearx()
                        print(("t =",0.01*j,":",f))
                eval(input())
            anywrong = False
            for i,c in enumerate(constraints):
                def f(v):
                    return c(v)
                def df(v):
                    c.setx(v)
                    retval = c.df_dx(v)
                    c.clearx()
                    return retval
                anywrong = anywrong or test_gradient(f,df,x,name="constraint %d %s"%(i,c.__class__.__name__))
            if anywrong:
                eval(input("Press enter to continue..."))

        #print "depth:",flatdepths,"predicted:",offsets,"with",len(inside),"in collision"
        if xmin is None:
            dxmin = None
        else:
            dxmin = xmin - x
        if xmax is None:
            dxmax = None
        else:
            dxmax = xmax - x

        if STEP_SIZE_METHOD == 'trust region':
            #build a box trust region
            n = len(dxdes)
            if xmin is None:
                dxmax = np.ones(n)*trust_region_size
                dxmin = -dxmax
            else:
                dxmin = np.maximum(dxmin,-np.ones(n)*trust_region_size)
                dxmax = np.minimum(dxmax,np.ones(n)*trust_region_size)

        #run the QP solve
        #depth[t] = d0 
        #depth[t+1] = A deltax + d0 with A = d depth/ dx
        #Solve for deltax to minimize ||W (deltax - dxdes)||^2 s.t. A deltax + d0 + A dxdes >= 0
        H = objective.hessian(x)
        if QP_SOLVE_METHOD == 'custom' or True:
            weight = objScoreWeight*H
        else:
            weight = H
        dx = cdata.solve_qp(W=weight,xdes=dxdes,A=rows,b=depths,xmin=dxmin,xmax=dxmax,verbose=verbose,**settings.qp_solver_params)
        
        dxnorm2 = np.dot(dx,dx)
        if dxnorm2 <= xepsilon2:
            if verbose>=2: print("Solved")
            res.status = 'local optimum'
            break
        if verbose>=2:
            if len(dx) > 10:
                print(("  Delta x norm",np.linalg.norm(dx),"with",len(np.nonzero(dx)[0]),"nonzero"))
            else:
                print(("  Delta x",dx,"norm",np.linalg.norm(dx)))
        if verbose >= 2:
            #test result
            if len(rows) > 0:
                print(("   Predicted depths:",np.dot(rows,dx) + depths))
        print("df_dT", np.size(df_dT), "dx", np.size(dx))

        if STEP_SIZE_METHOD == 'line search':

            H = objective.hessian(x)
            if QP_SOLVE_METHOD == 'custom' or True:
                weight = objScoreWeight*H
            else:
                weight = H
            dx = cdata.solve_qp(W=weight,xdes=dxdes,A=rows,b=depths,xmin=dxmin,xmax=dxmax,verbose=verbose,**settings.qp_solver_params)
            if np.isscalar(objective.hessian(x)):
                df = - np.matmul(np.identity(len(dx)) * objective.hessian(x), objective.minstep(x))
            else:
                df = - np.matmul(np.diag(objective.hessian(x)), objective.minstep(x))
            
            c_penalty = scoring_metric2(depths)
            #print 'depths',depths
            #print 'c_penalty',c_penalty

            if c_penalty != 0:
                penalty_parameter_tmp = vectorops.dot(df,dx)/((1 - 0.6) * c_penalty) #(1-rau)*penalty_parameter
            if penalty_parameter_tmp > penalty_parameter:
                penalty_parameter = penalty_parameter_tmp
            #print 'rau',penalty_parameter
            
            sorig = res.fx + penalty_parameter * c_penalty
            score_orig_trace.append(sorig)
            
            D = vectorops.dot(df, dx) - penalty_parameter * c_penalty

            alpha = 1.0
            
#            if xmin is not None:
#                #limit alpha by bounds
#                for i in range(len(x)):
#                    if xmin[i] == xmax[i]:
#                        dx[i] = 0
#                    elif x[i] == xmax[i] and dx[i] > 0:
#                        #active set
#                        dx[i] = 0
#                    elif x[i] == xmin[i] and dx[i] < 0:
#                        #active set
#                        dx[i] = 0
#                    elif x[i] + alpha*dx[i] > xmax[i]:
#                        alpha = (xmax[i]-x[i])/dx[i]
#                        if alpha == 0:
#                            print("STOPPED ON UPPER BOUND",i,"dx =",dx[i])
#                    elif x[i] + alpha*dx[i] < xmin[i]:
#                        alpha = (xmin[i]-x[i])/dx[i]
#                        if alpha == 0:
#                            print("STOPPED ON LOWER BOUND",i,"dx =",dx[i])
            
            xnext = objective.integrate(x,dx*alpha)
            
            if iters==0: 
                gx_trace.append(res.gx0)
            if verbose >= 2: print(("  Beginning line search at score",sorig,"... fx =",res.fx,", gx =",res.gx))
            #Now do a line search to optimize the residuals for *all* constraints not just the active ones
            dxnorm = math.sqrt(dxnorm2)
            alphaStall = min(xepsilon / dxnorm,alpha)
            alpha0 = alpha
            sfirst=None
            fxfirst=None
            gxfirst=None
            while alpha >= alphaStall:
                t1 = time.time()
                fxnew = objective.value(xnext)
                gxnew = np.array([c(xnext) for c in constraints])
                snew = fxnew + penalty_parameter * scoring_metric2(gxnew)
                t2 = time.time()
                tgeom += t2-t1
                res.x = xnext
                if sfirst is None:
                    sfirst,fxfirst,gxfirst = snew,fxnew,gxnew
                #print "   Alpha",alpha,"score",snew,"depths",res.gx
                if snew >= sorig + 0.01 * alpha * D or np.min(gxnew) < minimum_constraint_value:
                    alpha *= 0.6
                    xnext = objective.integrate(x,dx*alpha)
                else:
                    #decrease in score
                    break
            

                
            if alpha < alphaStall:
                if verbose>=1:
                    print(("  Line search stalled at step size",dxnorm,"score",snew,"... fx =",fxnew,", gx =",gxnew))
                    print(("     original step alpha =",alpha0,"score",sfirst,"... fx =",fxfirst,", gx =",gxfirst))
                    print(("     objective score weight",objScoreWeight))
                if objScoreWeight < xepsilon*10:
                    break
                xnext = x
            else:
                #print "  Completed line search at alpha =",alpha,"residual",newResidual,"...",nextresiduals
                if verbose>=2: print(("  Completed line search at alpha =",alpha,"score",snew,"... fx = ",res.fx,", gx =",res.gx))
        else:
            assert STEP_SIZE_METHOD == 'trust region',"Can only do line search or trust region now"
            dxnorm = math.sqrt(dxnorm2)
            xnext = objective.integrate(x,dx)
            sorig = res.fx*objScoreWeight + scoring_metric2(depths)
            score_orig_trace.append(sorig)
            if iters==0: 
                gx_trace.append(res.gx0)
            #Now do a line search to optimize the residuals for *all* constraints not just the active ones
            t0 = time.time()
            fxnew = objective.value(xnext)
            gxnew = np.array([c(xnext) for c in constraints])
            snew = fxnew*objScoreWeight + scoring_metric2(gxnew)
            t1 = time.time()
            tgeom += t1-t0
            accept_step = True
            reject_reason = None
            if snew >= sorig:
                accept_step = False
                reject_reason = 'increase in score'
                if verbose >= 2: print(("  Beginning step at score",sorig,"... fx =",res.fx,", gx =",res.gx))
                if verbose >= 2: print(("  Ending step at score",snew,"... fx =",fxnew,", gx =",gxnew))
                if verbose >= 2: print(("  Objective score weight",objScoreWeight))
                if verbose >= 2:
                    for i in range(15):
                        alpha = (i-5)*0.1
                        xnext = objective.integrate(x,dx*alpha)
                        dpred = np.dot(rows,dx*alpha) + depths
                        dvals = [c(xnext) for c in constraints]
                        fx = objective.value(xnext)
                        print(("  alpha",alpha,"f(x)",fx,"pred",min(dpred),"actual",min(sum(dvals,[])),"score",fx*objScoreWeight + scoring_metric([d for d in dvals if d < 0])))
                    eval(input("Press enter to continue > "))
            if np.min(gxnew) < minimum_constraint_value:
                reject_reason = 'under minimum constraint value'
            if not accept_step:
                xnext = x
                fxnew = res.fx
                gxnew = res.gx
                trust_region_size *= 0.5
                if trust_region_size > dxnorm:
                    trust_region_size = np.max(np.abs(dx))*0.5
                if verbose >=1: print(("  Step length %g rejected due to %s, shrinking trust region to %g"%(dxnorm,reject_reason,trust_region_size)))
            else:
                trust_region_size *= 5.0/2.0
                if verbose >=1: print(("  Step length %g accepted, growing trust region to %g"%(dxnorm,trust_region_size)))
        x = xnext
        res.trace.append(x)
        score_after_trace.append(snew)
        res.fx = fxnew
        res.gx = gxnew
        gx_trace.append(res.gx[:])
        #as min(gx) gets more negative, the objective score weight should decrease
        #as min(gx) gets more positive, the objective score weight should increase
        if (STEP_SIZE_METHOD == 'line search' and alpha < alphaStall):
            scale = 0.5
        else:
            maxviolation = min(res.gx)
            scale = 0.75*((math.atan(maxviolation*3)*2/math.pi + 0.5)*1.75 + 0.25)
        #print "Max violation",maxviolation,"scaling score weight by",scale
        objScoreWeight *= scale
        if objScoreWeight < settings.minimum_objective_score_weight:
            objScoreWeight = settings.minimum_objective_score_weight
        #if verbose>=2: raw_input()


    tfinish = time.time()
    res.x = x
    res.num_iterations = iters
    res.time = tfinish-tstart
    res.time_cons = tgeom
    if verbose>=1: print(("Completed in time",tfinish-tstart,"and",iters,"iterations"))
    if verbose>=1: print(("   geometry time",tgeom))
    if verbose>=1: print(("   final values f(x) =",res.fx,"g(x) =",res.gx))
    if verbose>=1 and min(res.gx) < -0.005:
        print("INFEASIBLE RESULT")
        print("TRACE: score\tf(x)\tg(x) intermediate\tg(x) overall")
        for i,x in enumerate(res.trace):                
            gx = [c(x) for c in constraints]
            if i == 0:
                print(("   \t\t",objective.value(x),'\t',gx_trace[i],'\t',gx))
            else:
                print(("   ",score_orig_trace[i-1],'->',score_after_trace[i-1],"\t",objective.value(x),'\t',gx_trace[i],'\t',gx))
    return res

def optimizeKlampt(objective,constraints,xinit,xmin=None,xmax=None,settings=None,verbose=1):
    """ Uses the Klamp't optimize solver to solve an NLP.  (Not used.)"""
    p = optimize.OptimizationProblem()
    def df(x):
        objective.setx(x)
        res = objective.df_dx(x)
        objective.clearx()
        return res
    p.setObjective(lambda x:objective(x),df)
    for c in constraints:
        def dc(x):
            c.setx(x)
            res = -c.df_dx(x)
            c.clearx()
            return res
        p.addInequality(lambda x:-c(x),dc)
    if xmin is not None:
        p.setBounds(xmin,xmax)
    solver = optimize.LocalOptimizer()
    solver.setSeed(xinit)
    if settings == None:
        settings = SemiInfiniteOptimizationSettings()
    (success,result) = solver.solve(p,numIters=settings.max_iters,tol=settings.xepsilon)
    res = SemiInfiniteOptimizationResult()
    res.x0 = xinit
    res.x = result
    res.trace = [xinit,result]
    return res

def optimizeSemiInfinite(objective,constraints,xinit,xmin=None,xmax=None,settings=None,verbose=1):
    """Solves a semi-infinite optimization problem of the form:

       min_x objective(x)
       s.t. c(x,y) >= 0 for all y in domain(c)
       [optional]
       xmin <= x <= xmax

    Returns a SemiInfiniteOptimizationResult.
    """
    #test whether we need to actually do semi-infinite optimization
    any_si = any(isinstance(c,SemiInfiniteConstraintInterface) for c in constraints)
    all_si = all(isinstance(c,SemiInfiniteConstraintInterface) for c in constraints)
    if not any_si:
        #no semi-infinite constraints
        return optimizeStandard(objective,constraints,xinit,xmin,xmax,settings,verbose)
    if not all_si:
        #need to adapt some of the constraints
        constraints = [c if isinstance(c,SemiInfiniteConstraintInterface) else SemiInfiniteConstraintAdaptor(c) for c in constraints]

    res = SemiInfiniteOptimizationResult()
    res.x0 = xinit

    x = xinit
    res.trace = [xinit]
    if settings is None:
        settings = SemiInfiniteOptimizationSettings()
    objScoreWeight = settings.initial_objective_score_weight
    penalty_parameter = settings.initial_penalty_parameter
    res.fx0 = objective.value(x)
    res.gx0 = np.zeros(len(constraints))
    res.x = x
    res.fx = res.fx0
    res.gx = np.zeros(len(constraints))
    cdata = ConstraintGenerationData(constraints)
    cdata.exclusion_distance = settings.parameter_exclusion_distance
    cdata.constraint_inflation = settings.constraint_inflation
    cdata.constraint_drop_value = settings.constraint_drop_value
    res.instantiated_params = cdata.instantiated_params
    minimum_constraint_value = settings.minimum_constraint_value
    xepsilon2 = settings.xepsilon**2*len(x)
    xepsilon = math.sqrt(xepsilon2)
    score_orig_trace = []
    score_after_trace = []
    gx_trace = []
    update_oracle = True
    trust_region_size = 0.5

    tstart = time.time()
    tgeom = 0
    iters = 0
    for iters in range(settings.max_iters):
        dxdes  = np.asarray(objective.minstep(x))
        #discover constraint points
        depths = []
        rows = []
        k = 0
        for c in constraints:
            c.setx(x)
        if update_oracle:
            t1 = time.time()
            cdepths = cdata.oracle(x,update_x=False,method=ORACLE_METHOD)
            t2 = time.time()
            tgeom += t2-t1
        else:
            #disable new contact point detection
            t1 = time.time()
            cdepths = cdata.oracle(x,update_x=False,method=None)
            t2 = time.time()
            tgeom += t2-t1
            #re-enable for next iteration
            update_oracle = True
        #test calling the oracle only in line search / trust region step
        if iters % 100 != 0:
            update_oracle = False
        res.gx = np.zeros(len(constraints))
        for i,c in enumerate(constraints):
            if len(cdepths[i]) > 0:
                res.gx[i] = min(cdepths[i])
            else:
                res.gx[i] = float('inf')
            if iters == 0:
                res.gx0[i] = res.gx[i]
                if res.gx0[i] < settings.minimum_constraint_value:
                    print(("WARNING: initial constraint value %d is below the minimum %g < %g"%(i,res.gx0[i],settings.minimum_constraint_value)))
                    minimum_constraint_value = res.gx0[i]
                    if verbose >= 1:
                        eval(input("Press enter to continue > "))
            if INCLUDED_CONSTRAINT_PARAMETERS == 'deepest':
                if len(cdepths[i]) > 0:
                    (d,y) = min(list(zip(cdepths[i],cdata.instantiated_params[i])))
                    df_dT = c.df_dx(x,y)
                    depths.append(d)
                    rows.append(df_dT)
                    k += 1
            else:
                #include all instantiated params
                for d,y in zip(cdepths[i],cdata.instantiated_params[i]):
                    df_dT = c.df_dx(x,y)
                    depths.append(d)
                    rows.append(df_dT)
                    k += 1
        for c in constraints:
            c.clearx()
        if iters == 0:
            if verbose>=1: print(("Beginning collision-free optimization at f(x) =",res.fx0,"g(x) =",res.gx0))

        #TEST GRADIENTS
        if DEBUG_GRADIENTS and iters == 1:
            if verbose >= 2:
                for i,c in enumerate(constraints):
                    for y in cdata.instantiated_params[i]:
                        c.setx(x)
                        f0 = c.value(x,y)
                        retval = c.df_dx(x,y)
                        c.clearx()
                        print(("CONSTRAINT GRADIENT, PARAM",y,"MAGNITUDE",np.linalg.norm(retval)))
                        print(("t = 0 :",f0))
                        for j in range(1,11):
                            xnext = objective.integrate(x,np.asarray(retval)*0.01*j)
                            c.setx(xnext)
                            f = c.value(x,y)
                            c.clearx()
                            print(("t =",0.01*j,":",f))
                eval(input())
            anywrong = False
            for i,c in enumerate(constraints):
                for y in cdata.instantiated_params[i]:
                    def f(v):
                        return c(v,y)
                    def df(v):
                        c.setx(v)
                        retval = c.df_dx(v,y)
                        c.clearx()
                        return retval
                    anywrong = anywrong or test_gradient(f,df,x,name="constraint %d %s"%(i,c.__class__.__name__))
            if anywrong:
                eval(input("Press enter to continue..."))

        #print "depth:",flatdepths,"predicted:",offsets,"with",len(inside),"in collision"
        if xmin is None:
            dxmin = None
        else:
            dxmin = xmin - x
        if xmax is None:
            dxmax = None
        else:
            dxmax = xmax - x

        if STEP_SIZE_METHOD == 'trust region':
            #build a box trust region
            n = len(dxdes)
            if xmin is None:
                dxmax = np.ones(n)*trust_region_size
                dxmin = -dxmax
            else:
                dxmin = np.maximum(dxmin,-np.ones(n)*trust_region_size)
                dxmax = np.minimum(dxmax,np.ones(n)*trust_region_size)

        #run the QP solve
        #depth[t] = d0 
        #depth[t+1] = A deltax + d0 with A = d depth/ dx
        #Solve for deltax to minimize ||W (deltax - dxdes)||^2 s.t. A deltax + d0 + A dxdes >= 0
        H = objective.hessian(x)
        if QP_SOLVE_METHOD == 'custom' or True:
            weight = objScoreWeight*H
        else:
            weight = H
        dx = cdata.solve_qp(W=weight,xdes=dxdes,A=rows,b=depths,xmin=dxmin,xmax=dxmax,verbose=verbose,**settings.qp_solver_params)
        if DEBUG_GRADIENTS and verbose >= 2:
            print("GRADIENT AGAINST CONSTRAINT:")
            k = 0
            for i,c in enumerate(constraints):
                for y in cdata.instantiated_params[i]:
                    c.setx(x)
                    f0 = c.value(x,y)
                    retval = c.df_dx(x,y)
                    c.clearx()
                    print(("  PARAM",y,"VALUE,",f0,"DIRECTIONAL DERIV",np.dot(retval,dx)))
                    if INCLUDED_CONSTRAINT_PARAMETERS == 'all':
                        if np.linalg.norm(np.asarray(retval)-rows[k]) > 1e-4:
                            print(("     DISCREPANCY IN DERIVATIVE",retval,"vs",rows[k]))
                    k += 1
            eval(input())
        dxnorm2 = np.dot(dx,dx)
        if dxnorm2 < xepsilon2:
            if verbose>=2: print("Solved")
            res.status = 'local optimum'
            break
        if verbose>=2:
            if len(dx) > 10:
                print(("  Delta x norm",np.linalg.norm(dx),"with",len(np.nonzero(dx)[0]),"nonzero"))
            else:
                print(("  Delta x",dx,"norm",np.linalg.norm(dx)))
        if verbose >= 2:
            #test result
            if len(rows) > 0:
                print(("   Predicted depths:",np.dot(rows,dx) + depths))

        if STEP_SIZE_METHOD == 'line search':
            
            H = objective.hessian(x)
            if QP_SOLVE_METHOD == 'custom' or True:
                weight = objScoreWeight*H
            else:
                weight = H
            dx = cdata.solve_qp(W=weight,xdes=dxdes,A=rows,b=depths,xmin=dxmin,xmax=dxmax,verbose=verbose,**settings.qp_solver_params)
            print('dx',dx,len(dx))
            print('dxdes',dxdes)
            #print 'Hessian', objective.hessian(x)
            
            if np.isscalar(objective.hessian(x)):
                df = - np.matmul(np.identity(len(dx)) * objective.hessian(x), objective.minstep(x))
                print('df',df ,len(df),len(objective.minstep(x)))
            else:
                df = - np.matmul(np.diag(objective.hessian(x)), objective.minstep(x))
                print('df',df,len(df),len(objective.minstep(x)))
            c_penalty = scoring_metric2(depths)
            #print 'depths',depths
            #print 'c_penalty',c_penalty

            if c_penalty != 0:
                penalty_parameter_tmp = vectorops.dot(df,dx)/((1 - 0.6) * c_penalty) #(1-rau)*penalty_parameter
            else:
                penalty_parameter_tmp = 0
            if penalty_parameter_tmp > penalty_parameter:
                penalty_parameter = penalty_parameter_tmp
            #print 'rau',penalty_parameter
            
            sorig = res.fx + penalty_parameter * c_penalty
            score_orig_trace.append(sorig)
            
            D = vectorops.dot(df, dx) - penalty_parameter * c_penalty

            alpha = 1.0
            
            """
            #limit by extrapolations of other points
            if len(inside) < len(cdata.instantiated_params):
                for i in xrange(len(rows)):
                    if depths[i] > 0:
                        dpred = alpha*np.dot(rows[i],dx) + depths[i]
                        if dpred < 0:
                            alpha = -depths[i]/np.dot(rows[i],dx)
                            if verbose>=2: print "Limiting alpha to",alpha,"Initial depth",depths[i],"pred depth",dpred
                            assert alpha >= 0 and alpha <= 1.0
                if alpha < 1.0:
                    if verbose>=1: print "Limited alpha to",alpha
            """
            xnext = objective.integrate(x,dx*alpha)
            sorig = _score(res.fx,cdepths,objScoreWeight)
            score_orig_trace.append(sorig)
            if iters==0: 
                gx_trace.append(res.gx0)
            if verbose >= 2: print(("  Beginning line search at score",sorig,"... fx =",res.fx,", gx =",res.gx))
            #Now do a line search to optimize the residuals for *all* constraints not just the active ones
            dxnorm2 = np.dot(dx,dx)
            dxnorm = math.sqrt(dxnorm2)
            alphaStall = min(xepsilon / dxnorm,alpha)
            num_instantiations0 = sum(len(p) for p in cdata.instantiated_params)
            alpha0 = alpha
            sfirst=None
            fxfirst=None
            gxfirst=None
            while alpha >= alphaStall:
                t1 = time.time()
                snew,fxnew,gxnew = score(objective,constraints,xnext,cdata,objScoreWeight,discoverNew=DETECT_NEW_COLLISIONS_IN_STEP)
                t2 = time.time()
                tgeom += t2-t1
                res.x = xnext
                if sfirst is None:
                    sfirst,fxfirst,gxfirst = snew,fxnew,gxnew
                #print "   Alpha",alpha,"score",snew,"depths",res.gx
                if snew >= sorig + 0.01 * alpha * D or np.min(gxnew) < minimum_constraint_value:
                    if DETECT_NEW_COLLISIONS_IN_STEP:
                        num_instantiations = sum(len(p) for p in cdata.instantiated_params)
                        if num_instantiations > num_instantiations0 and np.min(gxnew) < 0:
                            #go back to the prior state and update the step direction
                            xnext = x
                            fxnew = res.fx
                            gxnew = res.gx
                            if verbose >= 2:
                                print(("  Detected new constraint parameter during line search on iter",iters,", re-solving..."))
                                for i,c in enumerate(constraints):
                                    if len(cdata.instantiated_params[i]) > 0:
                                        c.setx(xnext)
                                        newcvalues = [c.value(xnext,y) for y in cdata.instantiated_params[i]]
                                        c.clearx()
                                        c.setx(x)
                                        cvalues = [c.value(x,y) for y in cdata.instantiated_params[i]]
                                        c.clearx()
                                        print(("  old constraint values",cvalues))
                                        print(("  new constraint values",newcvalues))
                                eval(input())
                            update_oracle = False
                            break
                    alpha *= 0.5
                    xnext = objective.integrate(x,dx*alpha)
                else:
#                    if DETECT_NEW_COLLISIONS_IN_LINE_SEARCH:
#                        num_instantiations = sum(len(p) for p in cdata.instantiated_params)
#                        if num_instantiations > num_instantiations0:
#                            update_oracle = False
                    #decrease in score
                    break
            if alpha < alphaStall:
                if verbose>=1:
                    print(("  Line search stalled at step size",dxnorm,"score",snew,"... fx =",fxnew,", gx =",gxnew))
                    print(("     original step alpha =",alpha0,"score",sfirst,"... fx =",fxfirst,", gx =",gxfirst))
                    print(("     objective score weight",objScoreWeight))
                if verbose >= 2:
                    for i in range(15):
                        alpha = (i-5)*0.1
                        xnext = objective.integrate(x,dx*alpha)
                        dpred = np.dot(rows,dx*alpha) + depths
                        dvals = []
                        for i,c in enumerate(constraints):
                            c.setx(xnext)
                            dvals.append([c.value(xnext,y) for y in cdata.instantiated_params[i]])
                            c.clearx()
                        fx = objective.value(xnext)
                        print(("  alpha",alpha,"f(x)",fx,"pred",min(dpred),"actual",min(sum(dvals,[])),"score",_score(fx,dvals,objScoreWeight)))
                    eval(input("Press enter to continue > "))
                if objScoreWeight < xepsilon*10:
                    break
                xnext = x
            else:
                #print "  Completed line search at alpha =",alpha,"residual",newResidual,"...",nextresiduals
                if verbose>=2: print(("  Completed line search at alpha =",alpha,"score",snew,"... fx = ",res.fx,", gx =",res.gx))
        else:
            assert STEP_SIZE_METHOD == 'trust region',"Can only do line search or trust region now"
            dxnorm = math.sqrt(dxnorm2)
            xnext = objective.integrate(x,dx)
            sorig = _score(res.fx,cdepths,objScoreWeight)
            score_orig_trace.append(sorig)
            if iters==0: 
                gx_trace.append(res.gx0)
            #Now do a line search to optimize the residuals for *all* constraints not just the active ones
            num_instantiations0 = sum(len(p) for p in cdata.instantiated_params)
            t0 = time.time()
            snew,fxnew,gxnew = score(objective,constraints,xnext,cdata,objScoreWeight,discoverNew=DETECT_NEW_COLLISIONS_IN_STEP)
            t1 = time.time()
            tgeom += t1-t0
            accept_step = True
            reject_reason = None
            if snew >= sorig:
                accept_step = False
                reject_reason = 'increase in score'
                if verbose >= 2: print(("  Beginning step at score",sorig,"... fx =",res.fx,", gx =",res.gx))
                if verbose >= 2: print(("  Ending step at score",snew,"... fx =",fxnew,", gx =",gxnew))
                if verbose >= 2: print(("  Objective score weight",objScoreWeight))
                if verbose >= 2:
                    for i in range(15):
                        alpha = (i-5)*0.1
                        xnext = objective.integrate(x,dx*alpha)
                        dpred = np.dot(rows,dx*alpha) + depths
                        dvals = []
                        for i,c in enumerate(constraints):
                            c.setx(xnext)
                            dvals.append([c.value(xnext,y) for y in cdata.instantiated_params[i]])
                            c.clearx()
                        fx = objective.value(xnext)
                        print(("  alpha",alpha,"f(x)",fx,"pred",min(dpred),"actual",min(sum(dvals,[])),"score",_score(fx,dvals,objScoreWeight)))
                    eval(input("Press enter to continue > "))
            if np.min(gxnew) < minimum_constraint_value:
                reject_reason = 'under minimum constraint value'
            if DETECT_NEW_COLLISIONS_IN_STEP:
                num_instantiations = sum(len(p) for p in cdata.instantiated_params)
                if num_instantiations > num_instantiations0 and np.min(gxnew) < 0:
                    if verbose >=1: print("  Detected new collision during trust region proposal step")
                    update_oracle = False
                    accept_step = False
                    reject_reason = 'new deepest collision'
            if not accept_step:
                xnext = x
                fxnew = res.fx
                gxnew = res.gx
                trust_region_size *= 0.5
                if trust_region_size > dxnorm:
                    trust_region_size = np.max(np.abs(dx))*0.5
                if verbose >=1: print(("  Step length %g rejected due to %s, shrinking trust region to %g"%(dxnorm,reject_reason,trust_region_size)))
            else:
                trust_region_size *= 5.0/2.0
                if verbose >=1: print(("  Step length %g accepted, growing trust region to %g"%(dxnorm,trust_region_size)))
        x = xnext
        res.trace.append(x)
        score_after_trace.append(snew)
        res.fx = fxnew
        res.gx = gxnew
        gx_trace.append(res.gx[:])
        #as min(gx) gets more negative, the objective score weight should decrease
        #as min(gx) gets more positive, the objective score weight should increase
        if (STEP_SIZE_METHOD == 'line search' and alpha < alphaStall):
            scale = 0.5
        else:
            maxviolation = min(res.gx)
            scale = 0.75*((math.atan(maxviolation*3)*2/math.pi + 0.5)*1.75 + 0.25)
        #print "Max violation",maxviolation,"scaling score weight by",scale
        objScoreWeight *= scale
        if objScoreWeight < settings.minimum_objective_score_weight:
            objScoreWeight = settings.minimum_objective_score_weight
        #if verbose>=2: raw_input()


    tfinish = time.time()
    res.x = x
    res.num_iterations = iters
    res.time = tfinish-tstart
    res.time_cons = tgeom
    if verbose>=1 or tfinish-tstart > 1: print(("Completed in time",tfinish-tstart,"and",iters,"iterations"))
    if verbose>=1 or tfinish-tstart > 1: print(("   geometry time",tgeom))
    if verbose>=1: print(("   final values f(x) =",res.fx,"g(x) =",res.gx))
    if verbose>=1 and min(res.gx) < -0.005:
        print("INFEASIBLE RESULT")
        print("TRACE: score\tf(x)\tg(x) intermediate\tg(x) overall")
        for i,x in enumerate(res.trace):                
            gx = []
            for c,params in zip(constraints,res.instantiated_params):
                c.setx(x)
                gx.append(min([c.value(x,y) for y in params]))
                c.clearx()
                
            if i == 0:
                print(("   \t\t",objective.value(x),'\t',gx_trace[i],'\t',gx))
            else:
                print(("   ",score_orig_trace[i-1],'->',score_after_trace[i-1],"\t",objective.value(x),'\t',gx_trace[i],'\t',gx))
    return res

