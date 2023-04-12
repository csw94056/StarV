"""
identityXidentity Class (x * y, where x \in X, y \in Y, X and Y are star sets)
Sung Woo Choi, 04/11/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class IdentityXIdentity(object):
    """
    IdentityXIdentity Class for reachability
    Author: Sung Woo Choi
    Date: 04/11/2023

    """
    u_ = 1 # index for upper bound case
    l_ = 2 # index for lower bound case

    x_ = 1 # index for x-coordinate constraint
    y_ = 2 # index for y-coordinate constraint
    z_ = 3 # index for z-coordinate constraint

    dzx_ = 4 # grandient on x-coordinate
    dzy_ = 5 # grandient on y-coordinate

    iux_ = 6 # intersection line on x-coordinate for upper bound case
    iuy_ = 7 # intersection line on y-coordinate for upper bound case
    ilx_ = 8 # intersection line on x-coordinate for lower bound case
    ily_ = 9 # intersection line on y-coordinate for lower bound case
    
    z_max = 4   
    z_min = 1

    num_of_points = 4;

    @staticmethod
    def f(x, y):
        return x * y
    
    def gf(x, y):
        """Gradient of x*y"""
        return x, y