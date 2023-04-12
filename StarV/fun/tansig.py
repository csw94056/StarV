"""
TanSig Class (Hyperbolic tangent sigmoid transfer function or TanH function)
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

class TanSig(object):
    """
    TanSig Class for reachability
    Author: Sung Woo Choi
    Date: 04/08/2023

    """

    @staticmethod
    def f(x):
        return np.tanh(x)
    
    @staticmethod
    def df(x):
        """Derivative of tansig(x)"""
        return 1 - np.tanh(x)**2
    
    def d2f(x):
        return -2*TanSig.f(x) * TanSig.df(x)
    
    @staticmethod
    def getConstraints(l, u):
        """Gets tangent line constraints on upper bounds and lower bounds"""
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)

        map0 = np.where(l == u)[0]
        al[map0] = 0
        au[map0] = 0

        bl = np.zeros(n - len(map0))
        bu = np.zeros(n - len(map0))

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        au[map1] = dyl[map1]
        # constraint 2: y <= y'(u) * (x - u) + y(u)
        bu[map1] = dyu[map1]
        
        map2 = np.where((u <= 0) & (l != u))[0]
        # constraint 1: y >= y'(l) * (x - l) + y(l)
        al[map2] = dyl[map2]
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])
        # constraint 2: y >= y'(u) * (x - u) + y(u) 
        bl[map2] = dyu[map2]

        map3 = np.where((l < 0) & (u > 0))[0]
        m = np.minimum(dyl[map3], dyu[map3])
        # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
        al[map3] = m
        # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
        au[map3] = m

        Du = np.diag(au)
        Dl = np.diag(al)
        gl = yl - al*l
        gu = yu - au*u
        return map0, Dl, Du, gl, gu
    

    @staticmethod
    def getConstraints_primary(l, u):
        """Gets tangent line constraints on upper bounds and lower bounds"""
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)

        map0 = np.where(l == u)[0]
        al[map0] = 0
        au[map0] = 0

        bl = np.zeros(n - len(map0))
        bu = np.zeros(n - len(map0))

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        au[map1] = dyu[map1]
        
        
        map2 = np.where((u <= 0) & (l != u))[0]
        al[map2] = dyl[map2]
        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])

        map3 = np.where((l < 0) & (u > 0))[0]
        m = np.minimum(dyl[map3], dyu[map3])
        al[map3] = m
        au[map3] = m

        Du = np.diag(au)
        Dl = np.diag(al)
        gl = yl - al*l
        gu = yu - au*u
        return map0, Dl, Du, gl, gu
    
    def getConstraints_secondary(l, u):
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)
        gl = np.zeros(n)
        gu = np.zeros(n)

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 2: y <= y'(u) * (x - u) + y(u)
        au[map1] = dyl[map1]
        gu[map1] = yl[map1] - au[map1] *l[map1] 
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] / l[map1])
        gl[map1] = yl[map1]  - al[map1] *l[map1] 

        map2 = np.where((u <= 0) & (l != u))[0]
        # constraint 2: y >= y'(u) * (x - u) + y(u) 
        al[map2] = dyu[map2]
        gl[map2] = yu[map2]  - al[map2] *u[map2] 

        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])
        gu[map2] = yu[map2] - au[map2] *u[map2] 


        map3 = np.where((l < 0) & (u > 0))[0]
        dmin = np.minimum(dyl[map3], dyu[map3])
        gux = (yu[map3] - dmin * u[map3]) / (1 - dmin)
        guy = gux 
        glx = (yl[map3] - dmin * l[map3]) / (1 - dmin)
        gly = glx

        mu = (yl[map3] - guy) / (l[map3] - gux)
        ml = (yu[map3] - gly) / (u[map3] - glx)

        al[map3] = mu
        au[map3] = ml
        gl[map3] = yl[map3] - al[map3]*l[map3]
        gu[map3] = yu[map3] - au[map3]*u[map3]

        Du = np.diag(au)
        Dl = np.diag(al)
        return Dl, Du, gl, gu

    def getConstraints_optimal(l, u):
        pass

    def reachApproxSparse(I, depthReduct=0, relaxFactor=0.0, lp_solver='gurobi'):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver, relaxFactor)
        l = l.reshape(N, 1)
        u = u.reshape(N, 1)

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl = TanSig.df(l)
        dyu = TanSig.df(u)

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

            gux = (yu_ - dmin * u_) / (1 - dmin)
            guy = gux 
            glx = (yl_ - dmin * l_) / (1 - dmin)
            gly = glx

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
            return TanSig.reachApproxSparse(I, depthReduct, relaxFactor, lp_solver)
        elif isinstance(I, Star):
            return TanSig.reachApproxStar(I, relaxFactor, lp_solver)
        else:
            raise Exception('error: unknown input set')