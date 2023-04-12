"""
Sparse Star Class
Sung Woo Choi, 04/03/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
import polytope as pc
import glpk
import gurobipy as gp
from gurobipy import GRB
from StarV.set.star import Star

class SparseStar(object):
    """
        SparseStar Class for reachability
        author: Sung Woo Choi
        date: 04/03/2023
        Representation of a SparseStar
        ===========================================================================================================================
        SparseStar set defined by

        ===========================================================================================================================
    """

    def __init__(self, *args):
        """
            Key Attributes:
            A = []; % basis matrix 
            C = []; % constraint matrix
            d = []; % constraint vector
            dim = 0; % dimension of the sparse star set
            nVars = 0; % number of predicate variables
            nZVars = 0; % number of non-basis predicate varaibles
            pred_lb = []; % lower bound of predicate variables
            pred_ub = []; % upper bound of predicate variables
        """

        len_ = len(args)
        if len_ == 6:
            [A, C, d, pred_lb, pred_ub, pred_depth] = copy.deepcopy(args)

            assert isinstance(A, np.ndarray), 'error: \
            basis matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            assert len(A.shape) == 2, 'error: \
            basis matrix should be a 2D numpy array'

            if len(d) > 0:
                assert isinstance(C, sp._csc.csc_matrix), 'error: \
                non-zero basis matrix should be a 2D scipy sparse csc matrix'
                assert isinstance(d, np.ndarray), 'error: \
                non-zero basis matrix should be a 2D numpy array'
                assert len(C.shape) == 2, 'error: \
                constraint matrix should be a 2D numpy array'
                assert len(d.shape) == 1, 'error: \
                constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], 'error: \
                inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], 'error: \
                inconsistent number of predicatve variables between constratint matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, 'error: \
            lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
            upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], 'error: \
            inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert pred_lb.shape[0] == pred_depth.shape[0], 'error: \
            inconsistent number of predicate variables between predicate bounds and predicate depth'

            self.A = A
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_depth = pred_depth
            self.dim = self.A.shape[0]
            if len(d) > 0:
                self.nVars = self.C.shape[1]
                self.nZVars = self.C.shape[1] + 1 - self.A.shape[1]
            else:
                self.nVars = self.dim
                self.nZVars = self.dim + 1 - self.A.shape[1]

        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)

        #     assert isinstance(lb, np.ndarray), 'error: \
        #     lower bound vector should be a 1D numpy array'
        #     assert isinstance(ub, np.ndarray), 'error: \
        #     upper bound vector should be a 1D numpy array'
        #     assert len(lb.shape) == 1, 'error: \
        #     lower bound vector should be a 1D numpy array'
        #     assert len(ub.shape) == 1, 'error: \
        #     upper bound vector should be a 1D numpy array'

        #     assert lb.shape[0] == ub.shape[0], 'error: \
        #         inconsistency between predicate lower bound and upper bound'
        #     if np.any(ub < lb):
        #         raise RuntimeError(
        #             "The upper bounds must not be less than the lower bounds for all dimensions")

        #     self.dim = lb.shape[0]
        #     nv = int(sum(ub > lb))
        #     c = 0.5 * (lb + ub)
        #     if self.dim == nv:
        #         v = np.diag(0.5 * (ub - lb))
        #     else:
        #         v = np.zeros((self.dim, nv))
        #         j = 0
        #         for i in range(self.dim):
        #             if ub[i] > lb[i]:
        #                 v[i, j] = 0.5 * (ub[i] - lb[i])
        #                 j += 1
        #     self.A = np.column_stack([c, v])

        #     # if dim > 3:
        #     #     self.A = torch.column_stack([c, torch.diag(v)]).to_sparse_csr()
        #     # else:
        #     #     self.A = torch.column_stack([c, torch.diag(v)])

        #     self.C = sp.csc_matrix([])
        #     self.d = np.array([])
        #     self.pred_lb = -np.ones(self.dim)
        #     self.pred_ub = np.ones(self.dim)
        #     self.pred_depth = np.zeros(self.dim)
        #     self.dim = self.dim
        #     self.nVars = self.dim
        #     self.nZVars = self.dim + 1 - self.A.shape[1]

        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
            assert isinstance(ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            assert len(lb.shape) == 1, 'error: \
            lower bound vector should be a 1D numpy array'
            assert len(ub.shape) == 1, 'error: \
            upper bound vector should be a 1D numpy array'

            assert lb.shape[0] == ub.shape[0], 'error: \
                inconsistency between predicate lower bound and upper bound'
            if np.any(ub < lb):
                raise RuntimeError(
                    "The upper bounds must not be less than the lower bounds for all dimensions")

            self.dim = lb.shape[0]
            nv = int(sum(ub > lb))
            self.A = np.zeros((self.dim, nv+1))
            j = 1
            for i in range(self.dim):
                if ub[i] > lb[i]:
                    self.A[i, j] = 1
                    j += 1

            self.C = sp.csc_matrix([])
            self.d = np.array([])
            self.pred_lb = lb
            self.pred_ub = ub
            self.pred_depth = np.zeros(self.dim)
            self.nVars = self.dim
            self.nZVars = self.dim + 1 - self.A.shape[1]

        elif len_ == 1:
            [P] = copy.deepcopy(args)

            assert isinstance(P, pc.Polytope), 'error: \
            input set is not a polytope Polytope'

            c = np.zeros([P.dim, 1])
            I = np.eye(P.dim)

            self.A = np.hstack([c, I])
            self.C = sp.csc_matrix(P.A)
            self.d = P.b
            self.dim = P.dim
            self.nVars = P.dim
            self.nZVars = 0
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
            self.pred_depth = np.zeros(self.dim)
            self.pred_lb, self.pred_ub = self.getRanges()
            
        elif len_ == 0:
            self.A = np.array([])
            self.C = sp.csc_matrix([])
            self.d = np.array([])
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
            self.pred_depth = np.array([])
            self.dim = 0
            self.nVars = 0
            self.nZVars = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 1, 2, 6)')

    def __str__(self, toDense = True):
        print('SparseStar Set:')
        print('A: \n{}'.format(self.A))
        if toDense:
            print('C_{}: \n{}'.format(self.C.getformat(), self.C.todense()))
        else:
            print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('pred_depth: {}'.format(self.pred_depth))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'

    def __repr__(self):
        print('SparseStar Set:')
        print('A: {}'.format(self.A.shape))
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('pred_depth: {}'.format(self.pred_depth.shape))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'

    def c(self, index=None):
        """Gets center vector of SparseStar"""
        if index is None:
            return self.A[:, 0].reshape(-1, 1)
        else:
            return self.A[index, 0].reshape(-1, 1)

    def X(self, row=None):
        """Gets basis matrix of predicate variables corresponding to the current dimension"""
        mA = self.A.shape[1]
        if row is None:
            return self.A[:, 1:mA]
        else:
            return self.A[row, 1:mA]

    def V(self, row=None):
        """Gets basis matrix"""
        mA = self.A.shape[1]
        if row is None:
            return np.column_stack([np.zeros((self.dim, self.nZVars)), self.X()])
        else:
            if isinstance(row, int) or isinstance(row, np.integer):
                return np.hstack([np.zeros(self.nZVars), self.X(row)])
            else:
                return np.column_stack([np.zeros((len(row), self.nZVars)), self.X(row)])

    def translation(self, v=None):
        """Translation of a sparse star: S = self + v"""
        if v is None:
            return copy.deepcopy(self)

        assert isinstance(v, np.ndarray), 'error: \
        offset vector should be an 1D numpy array'
        assert len(v.shape) == 1, 'error: \
        the translation vector should be a 1D numpy array'
        assert v.shape[0] == self.dim, 'error: \
        inconsistency between translation vector and SparseStar dimension'

        A = copy.deepcopy(self.A)
        A[:, 0] += v
        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def affineMap(self, W=None, b=None):
        """Affine mapping of a sparse star: S = W*self + b"""

        if W is None and b is None:
            return copy.deepcopy(self)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: \
                the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.dim, 'error: \
                inconsistency between mapping matrix and SparseStar dimension'

            A = np.matmul(W, self.A)

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: \
                the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: \
                offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: \
                    inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.dim, 'error: \
                    inconsistency between offset vector and SparseStar dimension'
                A = copy.deepcopy(self.A)

            A[:, 0] += b

        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def getMin(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """
        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V(index)
        if (f == 0).all():
            xmin = self.c(index, 0)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.nVars)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))
                res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

                if res.status == 0:
                    xmin = res.fun + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()
                # lp.interior()
                lp.simplex()
                # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.c(index)

            else:
                raise Exception(
                    'error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmin

    def getMax(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V(index)
        if (f == 0).all():
            xmax = self.c(index, 0)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.nVars)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))

            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))
                res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
                if res.status == 0:
                    xmax = -res.fun + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                # lp.interior()
                # default choice, interior may have a big floating point error
                lp.simplex()

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + self.V[index, 0]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax

    def getMins(self, map, lp_solver='gurobi'):
        n = len(map)
        xmin = np.zeros(n)
        for i in range(n):
            xmin[i] = self.getMin(map[i], lp_solver)
        return xmin

    def getMaxs(self, map, lp_solver='gurobi'):
        n = len(map)
        xmax = np.zeros(n)
        for i in range(n):
            xmax[i] = self.getMax(map[i], lp_solver)
        return xmax

    def estimateRange(self, index):
        """Estimates the minimum and maximum values of a state x[index]"""

        mA = self.A.shape[1]
        n = self.nVars
        p = n-mA+1

        l = self.pred_lb[p:n]
        u = self.pred_ub[p:n]

        pos_f = np.maximum(self.X(index), 0.0)
        neg_f = np.minimum(self.X(index), 0.0)

        xmin = self.c(index).flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c(index).flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def estimateRanges(self):
        """Estimates the lower and upper bounds of x"""

        mA = self.A.shape[1]
        n = self.nVars
        p = n-mA+1

        l = self.pred_lb[p:n]
        u = self.pred_ub[p:n]

        pos_f = np.maximum(self.X(), 0.0)
        neg_f = np.minimum(self.X(), 0.0)

        xmin = self.c().flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def getRange(self, index, lp_solver='gurobi'):
        """Gets the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(index)
        else:
            l = self.getMin(index)
            u = self.getMax(index)
            return l, u

    def getRanges(self, lp_solver='gurobi', relaxFactor=0.0):
        """Gets the lower and upper bound vectors of the state"""
        
        if relaxFactor == 0.0:
            if lp_solver == 'estimate':
                return self.estimateRanges()
            else:
                l = self.getMins(np.arange(self.dim))
                u = self.getMaxs(np.arange(self.dim))
                return l, u
        else:
            assert relaxFactor > 0.0 and relaxFactor < 1.0, 'error: \
            relaxation factor should be greater than 0.0 but less than 1.0'
            l, u = self.estimateRanges()
            n1 = round(1 - relaxFactor)*len(l)
            midx = np.argsort((u - l))[::-1]
            l1 = self.getMins(midx[0:n1])
            u1 = self.getMaxs(midx[0:n1])
            l[midx[0:n1]] = l1
            u[midx[0:n1]] = u1
            return l, u            
            
    def predReduction(self, p_map):
        """Reduces selected predicate variables"""

        assert (p_map >= 0).all() and (p_map < self.nVars).all(), 'error: \
        invalid predicate indexes, should be between {} and {}'.format(0, self.nVars)

        C = copy.deepcopy(self.C)
        d = copy.deepcopy(self.d)
        pred_lb = copy.deepcopy(self.pred_lb)
        pred_ub = copy.deepcopy(self.pred_ub)
        pred_depth = copy.deepcopy(self.pred_depth)

        pm = np.setdiff1d(np.arange(self.nVars), p_map)
        nC = C[:, pm].nonzero()[0]
        u = np.unique(nC)
        C = C[u, :]
        C = C[:, pm]
        d = d[u]
        pred_lb = pred_lb[u]
        pred_ub = pred_ub[u]
        pred_depth = pred_depth[u]
        return SparseStar(self.A, C, d, pred_lb, pred_ub, pred_depth)

    def depthReduction(self, d_max):
        """Reduces predicate variables based on the depth of predicate varibles"""

        assert d_max > 0, 'error: maximum allowed predicate variables depth should be greater than 0'

        max_depth = np.maximum(self.pred_depth)
        if d_max > max_depth:
            return copy.deepcopy(self)

        p_map = np.where(self.pred_depth >= d_max)
        return self.predReduction(p_map)

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True
        return res

    def minKowskiSum(self, S):
        """Minkowski Sum of two sparse stars"""

        assert isinstance(S, SparseStar), 'error: input is not a SparseStar'
        assert self.dim == S.dim, 'error: inconsistent dimension between the input and the self object'

        X = np.hstack((self.X(), S.X()))
        c = self.c() + S.c()
        A = np.hstack((c, X))

        OC1 = self.C[:, 0:self.nZVars]
        OC2 = self.C[:, self.nZVars:self.nVars]

        SC1 = S.C[:, 0:S.nZVars]
        SC2 = S.C[:, S.nZVars:S.nVars]

        C1 = sp.block_diag((OC1, SC1))
        C2 = sp.block_diag((OC2, SC2))

        if C1.nnz > 0:                    
            C = sp.hstack((C1, C2)).tocsc()
        else:
            C = C2.tocsc()
        d = np.concatenate((self.d, S.d))

        pred_lb = np.hstack((self.pred_lb[0:self.nZVars], S.pred_lb[0:S.nZVars],
                            self.pred_lb[self.nZVars:self.nVars], S.pred_lb[S.nZVars:S.nVars]))
        pred_ub = np.hstack((self.pred_ub[0:self.nZVars], S.pred_ub[0:S.nZVars],
                            self.pred_ub[self.nZVars:self.nVars], S.pred_ub[S.nZVars:S.nVars]))
        pred_depth = np.hstack((self.pred_depth[0:self.nZVars], S.pred_depth[0:S.nZVars],
                            self.pred_depth[self.nZVars:self.nVars], S.pred_depth[S.nZVars:S.nVars]))
        return SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)

    def concatenate(self, S):
        """Concatenates two sparse star sets """

        assert isinstance(S, SparseStar), 'error: input is not a SparseStar'

        c = np.concatenate(self.c(), S.c())
        X = block_diag(self.X(), S.X())
        A = np.hstack((c, X))

        OC1 = self.C[:, 0:self.nZVars]
        OC2 = self.C[:, self.nZVars:self.nVars]

        SC1 = S.C[:, 0:S.nZVars]
        SC2 = S.C[:, S.nZVars:S.nVars]

        C1 = sp.block_diag((OC1, SC1))
        C2 = sp.block_diag((OC2, SC2))

        if C1.nnz > 0:                    
            C = sp.hstack((C1, C2)).tocsc()
        else:
            C = C2.tocsc()
        d = np.concatenate((self.d, S.d))

        pred_lb = np.hstack((self.pred_lb[0:self.nZVars], S.pred_lb[0:S.nZVars],
                            self.pred_lb[self.nZVars:self.nVars], S.pred_lb[S.nZVars:S.nVars]))
        pred_ub = np.hstack((self.pred_ub[0:self.nZVars], S.pred_ub[0:S.nZVars],
                            self.pred_ub[self.nZVars:self.nVars], S.pred_ub[S.nZVars:S.nVars]))
        pred_depth = np.hstack((self.pred_depth[0:self.nZVars], S.pred_depth[0:S.nZVars],
                            self.pred_depth[self.nZVars:self.nVars], S.pred_depth[S.nZVars:S.nVars]))
        return SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)

    def sample(self, N):
        """Samples number of points in the feasible SparseStar set"""

        assert N >= 1, 'error: invalid number of samples'

        [lb, ub] = self.getRanges()

        V1 = np.array([])
        for i in range(self.dim):
            X = (ub[i] - lb[i]) * np.random.rand(2*N, 1) + lb[i]
            V1 = np.hstack([V1, X]) if V1.size else X

        V = np.array([])
        for i in range(2*N):
            v1 = V1[i, :]
            if self.contains(v1):
                V = np.vstack([V, v1]) if V.size else V1

        V = V.T
        if V.shape[1] >= N:
            V = V[:, 0:N]
        return V

    def toStar(self):
        if len(self.d) > 0:
            return Star(np.column_stack((self.c(), self.V())), self.C.todense(), self.d, self.pred_lb, self.pred_ub)
        else:
            return Star(np.column_stack((self.c(), self.V())), np.array([]), self.d, self.pred_lb, self.pred_ub)

    def contains(self, s):
        """
            Checks if a Star set contains a point.
            s : a star point (1D numpy array)
            
            return :
                1 -> a star set contains a point, s 
                0 -> a star set does not contain a point, s
                else -> error code from Gurobi LP solver
        """
        assert len(s.shape) == 1, 'error: invalid point. It should be 1D numpy array'
        assert s.shape[0] == self.dim, 'error: Dimension mismatch'     
        
        f = np.zeros(self.nVars)
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if self.pred_lb.size and self.pred_ub.size:
            x = m.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
        else:
            x = m.addMVar(shape=self.nVars)
        m.setObjective(f @ x, GRB.MINIMIZE)
        if len(self.d) > 0:
            C = self.C
            d = self.d
        else:
            C = sp.csr_matrix(np.zeros((1, self.nVars)))
            d = 0
        m.addConstr(C @ x <= d)
        Ae = sp.csr_matrix(self.V())
        be = s - self.c()
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))
        
    @staticmethod
    def rand(dim, N):
        """Generates a random SparseStar set
        """
        assert dim > 0, "error: Invalid dimension"

        A = np.random.rand(N, dim)
        # compute the convex hull
        P = pc.qhull(A)
        return SparseStar(P)

    @staticmethod
    def rand_bounds(dim):
        """Generate a random SparStar by random bounds"""

        assert dim > 0, 'error: invalid dimension'
        lb = -np.random.rand(dim)
        ub = np.random.rand(dim)
        return SparseStar(lb, ub)
