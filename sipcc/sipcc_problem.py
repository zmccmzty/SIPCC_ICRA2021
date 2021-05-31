import numpy as np
import scipy.linalg
import scipy.sparse
from .sip import ObjectiveFunctionInterface,ConstraintInterface,numeric_gradient,SemiInfiniteConstraintInterface,numeric_jacobian

class SIPCCOptimizationSettings:
    """Settings for the SIPCC programming solver."""
    def __init__(self):
        self.max_iters = 30
        self.mpcc_solver_params = {'Major_step_limit':0.02, 'Max_Iters_init':5}
        self.warm_start_z = True
        
class SIPCCProblem:
    """Formulates a semi-infinite problem with complementary constraints

    min_x,z f(x,z) s.t.               (objective)
        g(x,y) >= 0 for all y         (complementary)
        g(x;y)^T z(y) == 0 for all y  (complementary constraint)
        z(y) >= 0 for all y           (nonnegativity constraint)
        c(x) >= 0                     (x_ineq)
        d(x) == 0                     (x_eq)
        h(x,y,z) >= 0 for all y       (z_ineq)
        j(x,y,z) == 0 for all y       (z_eq)
        t(x,y,z) >= 0                 (xyz_ineq)
        s(x,y,z) == 0                 (xyz_eq)
    
    Here y is drawn from a domain D (domain)
    
    Attributes:
        objective (ObjectiveFunctionInterface or InfiniteObjectiveFunction):
        complementarity (SemiInfiniteConstraintInterface): the function g used in 
            the complementarity constraint
        domain (DomainInterface): the domain from which index points are drawn.
        x_ineq (ConstraintInterface): a constraint on x only.
        x_eq (ConstraintInterface): a constraint on x only.
        z_ineq (InfiniteConstraint): a constraint on x that must hold for all (y,z(y)).
        z_eq (InfiniteConstraint): a constraint on x that must hold for all (y,z(y)).
        xyz_ineq (InfiniteAggregateConstraint): a constraint on x and all (y,z(y))
            (typically a sum over z's).
        xyz_eq (InfiniteAggregateConstraint): a constraint on x and all (y,z(y))
            (typically a sum over z's).
        oracle (callable): a function \phi(x,y,z) that produces a list of new index
            points [y1,...,yk] that should be added to help the optimizer converge
            to a solution.
    """
    def __init__(self):
        self.objective = None
        self.complementarity = None
        self.domain = None
        self.x_ineq = None
        self.x_eq = None
        self.z_ineq = None
        self.z_eq = None
        self.xyz_ineq = None
        self.xyz_eq = None
        self.oracle = None
    
    def set_objective(self,f):
        """Sets the objective function"""
        if isinstance(f,ObjectiveFunctionInterface):
            self.objective = f
        elif isinstance(f,InfiniteObjectiveFunction):
            self.objective = f
        else:
            raise ValueError("f must be an ObjectiveFunctionInterface or InfiniteObjectiveFunction")

    def set_complementarity(self,g):
        assert isinstance(g,SemiInfiniteConstraintInterface)
        self.complementarity = g
        
    def add_ineq(self,f):
        """Adds an inequality on x, a single (x,y,z), or an aggregate
        infinite inequality constraint.
        """
        if isinstance(f,ConstraintInterface):
            if self.x_ineq is None:
                self.x_ineq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        elif isinstance(f,InfiniteConstraint):
            if self.z_ineq is None:
                self.z_ineq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        elif isinstance(f,InfiniteAggregateConstraint):
            if self.xyz_ineq is None:
                self.xyz_ineq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        else:
            raise ValueError("Constraint must be a ConstraintInterface, InfiniteConstraint, or InfiniteAggregateConstraint")
    
    def add_eq(self,f):
        """Adds an equality on x, a single (x,y,z), or an aggregate
        infinite equality constraint.
        """
        if isinstance(f,ConstraintInterface):
            if self.x_eq is None:
                self.x_eq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        elif isinstance(f,InfiniteConstraint):
            if self.z_eq is None:
                self.z_eq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        elif isinstance(f,InfiniteAggregateConstraint):
            if self.xyz_eq is None:
                self.xyz_eq = f
            else:
                raise NotImplementedError("TODO: concatenate constraints")
        else:
            raise ValueError("Constraint must be a ConstraintInterface, InfiniteConstraint, or InfiniteAggregateConstraint")
    
    def getNLP(self,x,y,z,x_lb=None,x_ub=None,complementarity_slack=0):
        """Returns a tuple (f,c,w_lb,w_ub,c_lb,c_ub) that defines an NLP over the
        stacked variable w=stack(x,z)
        
            min_w f(w) s.t.
            w_lb <= w <= w_ub
            c_lb <= c(w) <= c_ub
            
        which is equivalent to the SIPCC with the index points fixed at `y` and
        the complementarity slackness `complementarity_slack`.
        """
        inf = float('inf')
        if isinstance(self.objective,InfiniteObjectiveFunction):
            f = self.objective.stacked_objective(y)
        elif self.objective.variable == 'x':
            f = _SlicedXObjectiveFunction(self.objective,0,len(x))
        elif self.objective.variable == 'z':
            f = _SlicedZObjectiveFunction(self.objective,len(x))
        if x_lb == None:
            w_lb = np.hstack([np.full(len(x),-inf),np.zeros(len(z))])
        else:
            w_lb = np.hstack([np.asarray(x_lb),np.zeros(len(z))])
        if x_ub == None:
            w_ub = np.full(len(x)+len(z),inf)
        else:
            w_ub = np.hstack([np.asarray(x_ub),np.full(len(z),inf)])
        cs = []
        c_lb = []
        c_ub = []
        if self.complementarity:
            n = len(y)
            m = len(z)//n
            #Extract complementarity constraints
            # g(x,yi) >= 0
            d = max(self.complementarity.dims(),1)
            lb = np.zeros(d)
            ub = np.full(d,inf)
            for yi in y:
                # for i in range(m):
                cs.append(_SlicedXConstraintFunction(self.complementarity,0,len(x),yi))
                c_lb.append(lb)
                c_ub.append(ub)
            # g(x,y)^T z(y) < complementarity_slack
            cs.append(InfiniteConstraintToAggregateConstraint(InfiniteComplementarityConstraint(self.complementarity)).concatenated_constraint(x,y,z))
            # c_lb.append(np.full(len(y)*m,-inf))
            # c_ub.append(np.full(len(y)*m,complementarity_slack))
            c_lb.append(np.full(len(y),-inf))
            c_ub.append(np.full(len(y),complementarity_slack))
        if self.x_ineq:
            #Extract x inequality constraint
            cs.append(_SlicedXConstraintFunction(self.x_ineq,0,len(x)))
            d = max(self.x_ineq.dims(),1)
            c_lb.append(np.zeros(d))
            c_ub.append(np.full(d,inf))
        if self.x_eq:
            #Extract x equality constraint
            cs.append(_SlicedXConstraintFunction(self.x_eq,0,len(x)))
            d = max(self.x_eq.dims(),1)
            c_lb.append(np.zeros(d))
            c_ub.append(c_lb[-1])
        if self.z_ineq:
            #Extract z inequality constraint
            n = len(y)
            m = len(z)//n
            d = max(self.z_ineq.dims(),1)
            lb = np.zeros(d)
            ub = np.full(d,inf)
            for i in range(n):
                yi = y[i]
                cs.append(_SlicedXZConstraint(self.z_ineq,0,len(x),[yi],len(x)+i*m,len(x)+i*m+m))
                c_lb.append(lb)
                c_ub.append(ub)
        if self.z_eq:
            #Extract z equality constraint
            n = len(y)
            m = len(z)//n
            d = max(self.z_eq.dims(),1)
            lb = np.zeros(d)
            for i in range(n):
                yi = y[i]
                cs.append(_SlicedXZConstraint(self.z_eq,0,len(x),yi,len(x)+i*m,len(x)+i*m+m))
                c_lb.append(lb)
                c_ub.append(lb)
        if self.xyz_ineq:
            #Evaluate xyz constraint on stacked vector
            d = self.xyz_ineq.dims(y,z)
            cs.append(self.xyz_ineq.concatenated_constraint(x,y,z))
            c_lb.append(np.zeros(d))
            c_ub.append(np.full(d,inf))
        if self.xyz_eq:
            #Evaluate xyz constraint on stacked vector
            d = self.xyz_eq.dims(y,z)
            cs.append(self.xyz_eq.concatenated_constraint(x,y,z))
            c_lb.append(np.zeros(d))
            c_ub.append(c_lb[-1])
        return f,StackedConstraint(cs),w_lb,w_ub,np.concatenate(c_lb),np.concatenate(c_ub)
        
    def getQP(self,x0,y,z0,complementarity_slack=0):
        """Returns a tuple (P,q,A,w_lb,w_ub,c_lb,c_ub) defining a QP over the stacked
        variable w=stack(x,z) such that:
        
            min_w   w^T P w + q^T w s.t.
            w_lb <= w <= w_ub
            c_lb <= A w <= c_ub
        
        is the Taylor expansion of the SIPCC at (x0,y,z0)
        """
        inf = float('inf')
        if isinstance(self.objective,InfiniteObjectiveFunction):
            Hx = self.objective.hessian_x(x0,y,z0)
            Hz = self.objective.hessian_z(x0,y,z0)
            P = scipy.sparse.bmat([[Hx*0.5,None],[None,Hz*0.5]])
            gx = self.objective.gradient_x(x0,y,z0)
            gz = self.objective.gradient_z(x0,y,z0)
            q = np.concatenate([gx,gz])
        else:
            Hx = self.objective.hessian_x(x0,y,z0)
            Hz = scipy.sparse.csr_matrix((len(z0),len(z0)))
            P = scipy.sparse.bmat([[Hx*0.5,None],[None,Hz]])
            gx = self.objective.gradient_x(x0,y,z0)
            gz = np.zeros(len(z0))
            q = np.concatenate([gx,gz])
        w_lb = np.concatenate([np.full(len(x0),-inf),np.zeros(len(z0))])
        w_ub = np.full(len(x0)+len(z0),inf)
        As = []
        c_lb = []
        c_ub = []
        if self.complementarity:
            #Extract complementarity constraints
            d = max(self.complementarity.dims(),1)
            ub = np.full(d,inf)
            #complementarity function
            self.complementarity.setx(x0)
            for yi in y:
                cx0 = np.asarray(self.complementarity.value(x0,yi))
                Ax = self.complementarity.df_dx(x0,yi)
                As.append([Ax,None])
                c_lb.append(-cx0)
                c_ub.append(ub)
            self.complementarity.clearx()
            #complementarity condition g(x,y)^T z <= slack
            h = InfiniteComplementarityConstraint(self.complementarity)
            d = max(h.dims(),1)
            m = len(z0)//len(y)
            for i,yi in enumerate(y):
                zi = z0[i*m:i*m+m]
                c0 = np.asarray(h(x0,yi,zi))
                Ax = h.df_dx(x0,yi,zi)
                Az = h.df_dz(x0,yi,zi)
                As.append([Ax,Az])
                c_lb.append([-inf]*d)
                c_ub.append(complementarity_slack - c0)
        if self.x_ineq:
            #Extract x inequality constraint
            self.x_ineq.setx(x0)
            Ax = self.x_ineq.df_dx(x0)
            cx0 = np.asarray(self.x_ineq.value(x0))
            self.x_ineq.clearx()
            As.append([Ax,None])
            d = max(self.x_ineq.dims(),1)
            c_lb.append(-cx0)
            c_ub.append(np.full(d,inf))
        if self.x_eq:
            #Extract x equality constraint
            self.x_eq.setx(x0)
            Ax = self.x_eq.df_dx(x0)
            cx0 = self.x_eq.value(x0)
            self.x_eq.clearx()
            As.append([Ax,None])
            c_lb.append(-cx0)
            c_ub.append(c_lb[-1])
        if self.z_ineq:
            #Extract z inequality constraint
            n = len(y)
            m = len(z0)//n
            d = max(self.z_ineq.dims(),1)
            ub = np.full(d,inf)
            for i in range(n):
                yi = y[i]
                zi = z0[i*m:i*m+m]
                self.z_ineq.setxyz(x0,yi,zi)
                c0 = np.asarray(self.z_ineq.value(x0,yi,zi))
                Ax = self.z_ineq.df_dx(x0,yi,zi)
                Azi = self.z_ineq.df_dz(x0,yi,zi)
                self.z_ineq.clearxyz()
                components = []
                if i > 0:
                    components.append(scipy.sparse.csr_matrix((d,i*m)))
                components.append(Azi)
                if i*m+m < len(z0):
                    components.append(scipy.sparse.csr_matrix((d,len(z0)-(i*m+m))))
                Az = scipy.sparse.bmat(components)
                As.append([Ax,Az])
                c_lb.append(-c0)
                c_ub.append(ub)
        if self.z_eq:
            #Extract z equality constraint
            n = len(y)
            m = len(z0)//n
            d = max(self.z_eq.dims(),1)
            for i in range(n):
                yi = y[i]
                zi = z0[i*m:i*m+m]
                self.z_ineq.setxyz(x0,yi,zi)
                c0 = np.asarray(self.z_eq.value(x0,yi,zi))
                Ax = self.z_eq.df_dx(x0,yi,zi)
                Azi = self.z_eq.df_dz(x0,yi,zi)
                self.z_eq.clearxyz()
                components = []
                if i > 0:
                    components.append(scipy.sparse.csr_matrix((d,i*m)))
                components.append(Azi)
                if i*m+m < len(z0):
                    components.append(scipy.sparse.csr_matrix((d,len(z0)-(i*m+m))))
                Az = scipy.sparse.bmat(components)
                As.append([Ax,Az])
                c_lb.append(-c0)
                c_ub.append(c_lb[-1])
        if self.xyz_ineq:
            #Evaluate xyz constraint on stacked vector
            d = self.xyz_ineq.dims(y,z0)
            self.xyz_ineq.setxyz(x0,y,z0)
            c0 = np.asarray(self.xyz_ineq.value(x0,y,z0))
            Ax = self.xyz_ineq.df_dx(x0,y,z0)
            Az = self.xyz_ineq.df_dz(x0,y,z0)
            self.xyz_ineq.clearxyz()
            As.append([Ax,Az])
            c_lb.append(-c0)
            c_ub.append(np.full(d,inf))
        if self.xyz_eq:
            #Evaluate xyz constraint on stacked vector
            d = self.xyz_eq.dims(y,z0)
            self.xyz_eq.setxyz(x0,y,z0)
            c0 = np.asarray(self.xyz_eq.value(x0,y,z0))
            Ax = self.xyz_eq.df_dx(x0,y,z0)
            Az = self.xyz_eq.df_dz(x0,y,z0)
            self.xyz_eq.clearxyz()
            As.append([Ax,Az])
            c_lb.append(np.zeros(d))
            c_ub.append(c_lb[-1])
        return P,q,scipy.sparse.bmat(As),w_lb,w_ub,np.vstack(c_lb),np.vstack(c_ub)


class InfiniteObjectiveFunction:
    """Base class for an objective function f(x,y,z) that depends on index points y
    and variables z defined on those points.
    """
    def __init__(self):
        pass
    def value(self,x,y,z):
        """Returns the objective function value at x,y,z"""
        raise NotImplementedError()
    def gradient_x(self,x,y,z):
        """Returns the gradient of the objective function w.r.t. x at x,y,z.
        """
        raise NotImplementedError()
    def gradient_z(self,x,y,z):
        """Returns the gradient of the objective function w.r.t. z at x,y,z.
        """
        raise NotImplementedError()
    def hessian_x(self,x,y,z):
        """Compute / approximate the hessian of objective at x.

        A scalar return value is OK too, and is interpreted as a scaling of the identity matrix.

        A vector return value is OK too, and is interpreted as W=diag(hessian()).
        """
        raise NotImplementedError()
    def hessian_z(self,x,y,z):
        """Compute / approximate the hessian of objective at z.
        """
        raise NotImplementedError()
    def integrate_x(self,x,dx):
        """Implement non-Euclidean state spaces here. """
        return np.asarray(x)+np.asarray(dx)
    def stacked_objective(self,x,y,z):
        """Returns an ObjectiveFunctionInterface equivalent to this one over
        the stacked variable stack(x,z).
        """
        return _StackedInfiniteObjectiveAdaptor(self,x,y,z)

class InfiniteConstraint:
    """A infinite constraint is a function of the form h(x,y,z) ==0 or >= 0
    where x is in R^n, y is a single index point from some domain D,
    z is in R^m.

    To allow for caching between multiple constraints, to retrieve h(x,y,z) the methods setxyz,
    value, and clearxyz are called in the format:
      f.setxyz(x,y,z)
      fxyz = f.value(x,y,z)
      f.clearxyz()
      #result is in fxyz
    """
    def __init__(self):
        pass
    def dims(self):
        """Returns the number of dimensions of f(x).  0 indicates a scalar."""
        return 0
    def __call__(self,x,y,z):
        """Evaluates f(x,y,z)"""
        self.setxyz(x,y,z)
        fx = self.value(x,y,z)
        self.clearxyz()
        return fx
    def setxyz(self,x,y,z):
        """Called with the x,y,z value before value, df_dx, or df_dz can be called.
        Can set a cache here."""
        pass
    def clearxyz(self):
        """Called after all calls to value, df_dx, or df_dz with a given x,y,z.  Can clear a cache here."""
        pass
    def value(self,x,y,z):
        """Returns the function value at the optimization variable x, index points
        y, and index-specific variables z. x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dx(self,x,y,z):
        """Returns the Jacobian or gradient of the value with respect to x.
        x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dz(self,x,y,z):
        """Returns the Jacobian or gradient of the value with respect to z.
        x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,y,z,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearxyz()
        if self.dims()==0:
            res = numeric_gradient(lambda v:self.__call__(v,y,z),x,h)
        else:
            res = numeric_jacobian(lambda v:self.__call__(v,y,z),x,h)
        self.setxyz(x,y,z)
        return res
    def df_dz_numeric(self,x,y,z,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearxyz()
        if self.dims()==0:
            res = numeric_gradient(lambda v:self.__call__(x,y,v),z,h)
        else:
            res = numeric_jacobian(lambda v:self.__call__(x,y,v),z,h)
        self.setxyz(x,y,z)
        return res


class InfiniteAggregateConstraint:
    """A infinite aggregate constraint is a function of the form h(x,y,z) == 0
    or >= 0 where x is in R^n, y is a set of index points from some domain D,
    z is in R^(m|y|) with m a constant.

    To allow for caching between multiple constraints, to retrieve h(x,y,z) the methods setxyz,
    value, and clearxyz are called in the format:
      f.setxyz(x,y,z)
      fxyz = f.value(x,y,z)
      f.clearxyz()
      #result is in fxyz
    """
    def __init__(self):
        pass
    def dims(self,y,z):
        """Returns the number of dimensions of f(x,y,z).  0 indicates a scalar."""
        return 0
    def __call__(self,x,y,z):
        """Evaluates f(x,y,z)"""
        self.setxyz(x,y,z)
        fx = self.value(x,y,z)
        self.clearxyz()
        return fx
    def setxyz(self,x,y,z):
        """Called with the x,y,z value before value, df_dx, or df_dz can be called.
        Can set a cache here."""
        pass
    def clearxyz(self):
        """Called after all calls to value, df_dx, or df_dz with a given x,y,z.  Can clear a cache here."""
        pass
    def clearx(self):
        pass
    def value(self,x,y,z):
        """Returns the function value at the optimization variable x, index points
        y, and index-specific variables z. x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dx(self,x,y,z):
        """Returns the Jacobian or gradient of the value with respect to x.
        x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dz(self,x,y,z):
        """Returns the Jacobian or gradient of the value with respect to z.
        x,y,z must be previously set using setxyz."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,y,z,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearxyz()
        if self.dims(y,z)==0:
            res = numeric_gradient(lambda v:self.__call__(v,y,z),x,h)
        else:
            res = numeric_jacobian(lambda v:self.__call__(v,y,z),x,h)
        self.setxyz(x,y,z)
        return res
    def df_dz_numeric(self,x,y,z,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearxyz()
        if self.dims(y,z)==0:
            res = numeric_gradient(lambda v:self.__call__(x,y,v),z,h)
        else:
            res = numeric_jacobian(lambda v:self.__call__(x,y,v),z,h)
        self.setxyz(x,y,z)
        return res
    def concatenated_constraint(self,x,y,z):
        """Returns a ConstraintInterface on the concatenated vector (x,z) for 
        a fixed set of index points y.
        """
        return _StackedInfiniteConstraintAdaptor(self,y,len(x))

class InfiniteComplementarityConstraint(InfiniteConstraint):
    """Represents the constraint g(x,y)^T z <= 0 with g a
    SemiInfiniteConstraintInterface object.
    """
    def __init__(self,g):
        assert isinstance(g,SemiInfiniteConstraintInterface)
        self.g = g
    def dims(self):
        return 0
    def setxyz(self,x,y,z):
        self.g.setx(x)
    def value(self,x,y,z):
        g = self.g(x,y)
        return np.dot(np.ones(len(z))*g,z)
    def clearxyz(self):
        self.g.clearx()
    def df_dx(self,x,y,z):
        J_ = self.g.df_dx(x,y)
        J = np.repeat(J_.reshape((1,-1)),len(z),axis=0)
        return (J.T).dot(z)
    def df_dz(self,x,y,z):
        return np.asarray([self.g.value(x,y)]*len(z))
class InfiniteConstraintToAggregateConstraint(InfiniteAggregateConstraint):
    """Converts a single InfiniteConstraint into an InfiniteAggregateConstraint
    simply by stacking.
    """
    def __init__(self,f):
        self.f = f
    def clearx(self):
        pass
    def dims(self,y,z):
        n = len(y)
        return n*max(1,self.f.dims())
    def value(self,x,y,z):
        n = len(y)
        m = len(z)//n
        fs = []
        for i in range(n):
            yi = y[i]
            zi = z[i*m:i*m+m]
            fs.append(self.f(x,yi,zi))
        if not hasattr(fs[0],'__iter__'):
            return np.array(fs)
        return np.vstack(fs)
    def df_dx(self,x,y,z):
        n = len(y)
        m = len(z)//n
        dfs = []
        for i in range(n):
            yi = y[i]
            zi = z[i*m:i*m+m]
            dfs.append(scipy.sparse.csr_matrix(self.f.df_dx(x,yi,zi)))
        return scipy.sparse.vstack(dfs)
    def df_dz(self,x,y,z):
        n = len(y)
        m = len(z)//n
        dfs = []
        for i in range(n):
            yi = y[i]
            zi = z[i*m:i*m+m]
            dfs.append(self.f.df_dz(x,yi,zi))
        return scipy.sparse.block_diag(tuple(dfs))
    
class _StackedInfiniteObjectiveAdaptor(ObjectiveFunctionInterface):
    def __init__(self,inf_objective,y,xlen):
        self.inf_objective = inf_objective
        self.xlen = xlen
        self.y = y
    def setx(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen,:]
        self.inf_objective.setxyz(xi,self.y,zi)
    def value(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen,:]
        return self.inf_objective.value(xi,self.y,zi)
    def clearx(self):
        self.inf_objective.clearx()
    def gradient(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen,:]
        dfx = self.inf_objective.gradient_x(xi,self.y,zi)
        dfz = self.inf_objective.gradient_z(xi,self.y,zi)
        return np.concatenate([dfx,dfz])
    def hessian(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen,:]
        Hx = self.inf_objective.hessian_x(xi,self.y,zi)
        Hz = self.inf_objective.hessian_z(xi,self.y,zi)
        return scipy.sparse.bmat([[Hx,None],[None,Hz]])

class _StackedInfiniteConstraintAdaptor(ConstraintInterface):
    """Internal.  Evaluates a constraint on a stacked vector."""
    def __init__(self,inf_constraint,y,xlen):
        self.inf_constraint = inf_constraint
        self.y = y
        self.xlen = xlen
    def dims(self):
        return self.inf_constraint.dims()
    def setx(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen:]
        self.inf_constraint.setxyz(xi,self.y,zi)
    def value(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen:]
        return self.inf_constraint.value(xi,self.y,zi)
    def clearx(self):
        self.inf_constraint.clearx()
    def df_dx(self,x):
        xi = x[:self.xlen]
        zi = x[self.xlen:]
        dfx = self.inf_constraint.df_dx(xi,self.y,zi)
        dfz = self.inf_constraint.df_dz(xi,self.y,zi)
        if not scipy.sparse.issparse(dfx):
            dfx = scipy.sparse.csr_matrix(dfx)
        if not scipy.sparse.issparse(dfz):
            dfz = scipy.sparse.csr_matrix(dfz)
        return scipy.sparse.hstack([dfx,dfz])

class SliceConstraintFunction(ConstraintInterface):
    """A function f(x)=fslice(x[xstart:xend])
    """
    def __init__(self,fslice,xstart,xend):
        assert isinstance(fslice,ConstraintInterface)
        self.fslice = fslice
        self.xstart = xstart
        self.xend = xend
    def dims(self):
        return self.fslice.dims()
    def setx(self,x):
        self.fslice.setx(x[self.xstart:self.xend])
    def clearx(self):
        self.fslice.clearx()
    def value(self,x):
        return self.fslice.value(x[self.xstart:self.xend])
    def df_dx(self,x):
        dx = self.fslice.df_dx(x[self.xstart:self.xend])
        components = []
        if self.xstart > 0:
            components.append(scipy.sparse.csr_matrix((self.dims(),self.xstart)))
        components.append(dx)
        if self.xend < len(x):
            components.append(scipy.sparse.csr_matrix((self.dims(),len(x)-self.xend)))
        return scipy.sparse.bmat(components)

class StackedConstraint(ConstraintInterface):
    def __init__(self,fs):
        self.fs = fs
    def dims(self):
        return sum(max(f.dims(),1) for f in self.fs)
    def setx(self,x):
        for f in self.fs:
            f.setx(x)
    def clearx(self):
        for f in self.fs:
            f.clearx()
    def value(self,x):
        vs = [np.array(f.value(x)).reshape((-1,1)) for f in self.fs]
        return np.concatenate((vs))    
    def df_dx(self,x):
        dfs = [f.df_dx(x) for f in self.fs]
        for i,df in enumerate(dfs):
            if len(df.shape) == 1:
                dfs[i] = df.reshape((1,len(df)))
        return scipy.sparse.vstack(dfs)   
    
class _SlicedXObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x)=fslice(x[xstart:xend])
    """
    def __init__(self,fslice,xstart,xend):
        assert isinstance(fslice,ObjectiveFunctionInterface)
        self.fslice = fslice
        self.xstart = xstart
        self.xend = xend
    def value(self,x):
        return self.fslice.value(x[self.xstart:self.xend])
    def gradient(self,x):
        dx = self.fslice.gradient(x[self.xstart:self.xend])
        res = np.zeros(len(x))
        res[self.xstart:self.xend] = dx
        return res
    def hessian(self,x):
        Hx = self.fslice.hessian(x[self.xstart:self.xend])
        #TODO: sparse matrices
        H = np.zeros((len(x),len(x)))
        H[self.xstart:self.xend,self.xstart:self.xend] = Hx
        return H

class _SlicedZObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x)=fslice(x[xstart:xend])
    """
    def __init__(self,fslice,xstart):
        assert isinstance(fslice,ObjectiveFunctionInterface)
        self.fslice = fslice
        self.xstart = xstart
    def value(self,x):
        return self.fslice.value(x[self.xstart:])
    def gradient(self,x):
        dx = self.fslice.gradient(x[self.xstart:])
        res = np.zeros(len(x))
        res[self.xstart:] = dx
        return res
    def hessian(self,x):
        Hx = self.fslice.hessian(x[self.xstart:])
        #TODO: sparse matrices
        H = np.zeros((len(x),len(x)))
        H[self.xstart:,self.xstart:] = Hx
        return H
        
class _SlicedXConstraintFunction(ConstraintInterface):
    def __init__(self,fslice,xstart,xend,y):
        assert isinstance(fslice,SemiInfiniteConstraintInterface)
        self.fslice = fslice
        self.xstart = xstart
        self.xend = xend
        self.y = y
    def dims(self):
        return max(self.fslice.dims(),1)
    def setx(self,x):
        self.fslice.setx(x[self.xstart:self.xend])
    def clearx(self):
        self.fslice.clearx()
    def value(self,x):
        return self.fslice.value(x[self.xstart:self.xend],self.y)
    def df_dx(self,x):
        dx = self.fslice.df_dx(x[self.xstart:self.xend],self.y)
        components = []
        if self.xstart > 0:
            components.append(scipy.sparse.csr_matrix((self.dims(),self.xstart)))
        components.append(np.asarray(dx))
        if self.xend < len(x):
            components.append(scipy.sparse.csr_matrix((self.dims(),len(x)-self.xend)))
        return scipy.sparse.bmat([components])

class _SlicedXZConstraint(ConstraintInterface):
    def __init__(self,fslice,xstart,xend,y,zstart,zend):
        assert isinstance(fslice,InfiniteConstraint)
        self.fslice = fslice
        self.xstart = xstart
        self.xend = xend
        self.y = y
        self.zstart = zstart
        self.zend = zend
    def dims(self):
        return max(self.fslice.dims(),1)
    def setx(self,x):
        self.fslice.setxyz(x[self.xstart:self.xend],self.y,x[self.zstart:self.zend])
    def clearx(self):
        self.fslice.clearxyz()
    def value(self,x):
        return self.fslice.value(x[self.xstart:self.xend],self.y,x[self.zstart:self.zend])
    def df_dx(self,x):
        dx = self.fslice.df_dx(x[self.xstart:self.xend],self.y,x[self.zstart:self.zend])
        dz = self.fslice.df_dz(x[self.xstart:self.xend],self.y,x[self.zstart:self.zend])
        assert self.xend <= self.zstart 
        components = []
        if self.xstart > 0:
            components.append(scipy.sparse.csr_matrix((self.dims(),self.xstart)))
        components.append(dx)
        if self.xend < self.zstart:
            components.append(scipy.sparse.csr_matrix((self.dims(),self.zstart-self.xend)))
        components.append(dz)
        if self.zend < len(x):
            components.append(scipy.sparse.csr_matrix((self.dims(),len(x)-self.zend)))
        return scipy.sparse.bmat([components])