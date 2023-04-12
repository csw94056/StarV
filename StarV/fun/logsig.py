"""
TanSig Class (Log-sigmoid transfer function or Sigmoid function)
Sung Woo Choi, 04/08/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class LogSig(object):
    """
    LogSig Class for reachability   
    Author: Sung Woo Choi
    Date: 04/08/2023

    """

    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))
    
    def df(x):
        """Derivative of logsig(x)"""
        f = LogSig.f(x)
        return f * (1 - f)
    



    def reachApproxSparse(I, depthReduct=0, relaxFactor=0.0, lp_solver='gurobi'):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver, relaxFactor)
        l = l.reshape(N, 1)
        u = u.reshape(N, 1)

        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl = LogSig.df(l)
        dyu = LogSig.df(u)

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        new_A = np.hstack((np.zeros((N, 1)), A0))

        map1 = np.where(l == u)[0]
        if len(map1):
            new_A[map1, 0] = yl[map1]
            new_A[map1, 1:m+1] = 0

        nv = I.nVars + m

        ## l > 0 & l != u
        map1 = np.where(l[map0] >= 0)[0]
        if len(map1):
            map_ = map0[map1]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y <= y'(l) * (x - l) + y(l)
            C11 = sp.hstack((Z, -dyl_*I.X(map_), A0[map_, :]))
            d11 = dyl_*(I.c(map_) - l[map_]) + yl_

            # constraint 2: y <= y'(u) * (x - u) + y(u) 
            C12 = sp.hstack((Z, -dyu_*I.X(map_), A0[map_, :]))
            d12 = dyu_*(I.c(map_) - u[map_]) + yu_

            # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u[map_] - l[map_])
            C13 = sp.hstack((Z, g*I.X(map_), -A0[map_, :]))
            d13 = -g*(I.c(map_) - l[map_]) - yl_

            C1 = sp.vstack((C11, C12, C13)).tocsc()
            d1 = np.vstack((d11, d12, d13)).flatten()
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))

        ## u <= 0 & l != u
        map1 = np.where(u[map0] <= 0)[0]
        if len(map1):
            map_ = map0[map1]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C21 = sp.hstack((Z, dyl_*I.X(map_), -A0[map_, :]))
            d21 = -dyl_*(I.c(map_) - l[map_]) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = sp.hstack((Z, dyu_*I.X(map_), -A0[map_, :]))
            d22 = -dyu_*(I.c(map_) - u[map_]) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u[map_] - l[map_])
            C23 = sp.hstack((Z, -g*I.X(map_), A0[map_, :]))
            d23 = g*(I.c(map_) - l[map_]) + yl_

            C2 = sp.vstack((C21, C22, C23)).tocsc()
            d2 = np.vstack((d21, d22, d23)).flatten()
        else:
            C2 = sp.csc_matrix((0, nv))
            d2 = np.empty((0))

        map1 = np.where((l < 0) & (u > 0))[0]
        if len(map1):
            l_ = l[map1]
            u_ = u[map1]
            yl_ = yl[map1]
            yu_ = yu[map1]
            dyl_ = dyl[map1]
            dyu_ = dyu[map1]

            dmin = np.minimum(dyl_, dyu_)
            Z = sp.csc_matrix((len(map1), I.nZVars))

            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            C31 = sp.hstack((Z, dmin*I.X(map1), -A0[map1, :]))
            d31 = -dmin*(I.c(map1) - l_) - yl_

            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
            C32 = sp.hstack((Z, -dmin*I.X(map1), A0[map1, :]))
            d32 = dmin*(I.c(map1) - u_) + yu_

            gux = (yu_ - dmin * u_ - 0.5) / (0.25 - dmin)
            guy = 0.25 * gux + 0.5 
            glx = (yl_ - dmin * l_ - 0.5) / (0.25 - dmin)
            gly = 0.25 * glx + 0.5

            mu = (yl_ - guy) / (l_ - gux)
            ml = (yu_ - gly) / (u_ - glx)
            
            # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
            C33 = sp.hstack((Z, ml*I.X(map1), -A0[map1, :]))
            d33 = -ml*(I.c(map1) - u_) - yu_

            # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
            C34 = sp.hstack((Z, -mu*I.X(map1), A0[map1, :]))
            d34 = mu*(I.c(map1) - l_) + yl_

            C3 = sp.vstack((C31, C32, C33, C34)).tocsc()
            d3 = np.vstack((d31, d32, d33, d34)).flatten()
        else:
            C3 = sp.csc_matrix((0, nv))
            d3 = np.empty((0))

        n = I.C.shape[0]
        if len(I.d):
            C0 = sp.hstack((I.C, sp.csc_matrix(n, m))) 
            d0 = I.d
        else:
            C0 = sp.csc_matrix((0, I.nVars+m))
            d0 = np.empty((0))

        new_C = sp.vstack((C0, C1, C2, C3))
        new_d = np.hstack((d0, d1, d2, d3))

        new_pred_lb = np.hstack((I.pred_lb, yl[map0].flatten()))
        new_pred_ub = np.hstack((I.pred_ub, yu[map0].flatten()))
        pd1 = I.pred_depth + 1
        new_pred_depth = np.hstack((pd1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if depthReduct > 0:
            S = S.depthReuction(depthReduct)
        return S
    
    def reach(I, depthReduct=0, relaxFactor=0.0, lp_solver='gurobi', pool=None):
        if isinstance(I, SparseStar):
            return LogSig.reachApproxSparse(I, depthReduct, relaxFactor, lp_solver)
        elif isinstance(I, Star):
            return LogSig.reachApproxStar(I, relaxFactor, lp_solver)
        else:
            raise Exception('error: unknown input set')
