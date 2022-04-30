# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:58:31 2022

@author: mason
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:22:12 2022

@author: mason
"""

from choi import *
import numpy as np


import cvxpy as cp
from math import log



def optimization(d,q,channel):

    r1 = cp.Variable(1)
    r2 = cp.Variable(1)
    r3 = cp.Variable(1)
    r4 = cp.Variable(1)

    mu = cp.Variable(1)
    ident = np.identity(d**2)
    swapp = swap(d)
    gammma = gamma(d)

    constraints = [r1 * ident + r2 * swapp >> 0]
    constraints += [r3 * ident + r4 * swapp >> 0]
    constraints += [((r1-r3) * ident + (r2-r4) * gammma) >> choi(d,q,channel)]
    constraints += [mu >= d * r1 + r2 + d * r3 + r4]

    prob = cp.Problem(cp.Minimize(mu), constraints)

    prob.solve()

    return log(prob.value,2)

print(optimization(2,1,depol))

