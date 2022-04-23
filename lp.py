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


import cvxpy as cp
from math import log



def optimization(d,q,channel):

    r1 = cp.Variable()
    r2 = cp.Variable()
    r3 = cp.Variable()
    r4 = cp.Variable()

    mu = cp.Variable()
    identity = np.identity(d**2)

    constraints = [r1 * identity + r2 * swap(d) >> 0]
    constraints += [r3 * identity + r4 * swap(d) >> 0]
    constraints += [(r1-r3) * identity + (r2-r4) * gamma(d) >> choi(d,q,channel)]
    constraints += [r1 + r2 + r3 + r4 <= mu]

    

    prob = cp.Problem(cp.Minimize(mu), constraints)

    prob.solve()

    return log(prob.value,2)

print(optimization(2,1,depol))

