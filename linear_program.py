import numpy as np
import cvxpy as cp
from choi import *
from numpy import log2
import matplotlib.pyplot as plt
import time
import choi_plot as cplot


# def phi(dimension):
#     d = dimension
#     arr = np.zeros((d ** 2, d ** 2))
#     indices = []
#     for i in range(d):
#         indices.append(0 + i * (d + 1))
#     for j in indices:
#         for k in indices:
#             arr[j][k] = 1
#     return arr

# plots the capacity bound for depolarizing/ werner channel with a given d
def plot_lp_depol(dimension):
    start = time.time()
    d = dimension
    c = []

    q_vals = np.arange(0, d ** 2 / (d ** 2 - 1), 0.01, dtype=float)

    r1 = cp.Variable(1)
    r2 = cp.Variable(1)
    r3 = cp.Variable(1)
    r4 = cp.Variable(1)
    mu = cp.Variable()

    for q in q_vals:
        constraints = [r1 >= r2, r3 >= r4]
        constraints += [r1 + r2 >= 0, r3 + r4 >= 0]
        constraints += [(d * (r1 + r3) + r2 + r4) <= mu]
        constraints += [(r1 - r3 - q / d + d * (r2 - r4) - d * (1 - q)) >= 0]
        constraints += [(r1 - r3 - q / d) >= 0]
        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve()
        c.append(log2(prob.value))

    plt.plot(q_vals, c, label="d = " + str(d))
    plt.legend(loc="upper left")

    plt.title('Depolarizing Channel with d=' + str(d))
    plt.xlabel('q-parameter')
    plt.ylabel('quantum capacity')
    plt.savefig('./plot/depol/lp_depol_' + str(d) + '.png')

    # plt.show()
    end = time.time()
    print(end - start)


# print(plot_lp_depol(3))


# plot a general plot of capacity bound of depolar/werner channels with a given lower/upper d
def plot_all_lp_depol(upper):
    start = time.time()
    listt = [2, 10]
    listt += list(np.arange(20, upper+1, 20))
    for i in (listt):
        d = i
        c = []
        t = np.arange(0, 1, 0.01, dtype=float)
        q_vals = t * (d ** 2 / (d ** 2 - 1))

        r1 = cp.Variable(1)
        r2 = cp.Variable(1)
        r3 = cp.Variable(1)
        r4 = cp.Variable(1)
        mu = cp.Variable()

        for q in q_vals:
            constraints = [r1 >= r2, r3 >= r4]
            constraints += [r1 + r2 >= 0, r3 + r4 >= 0]
            constraints += [(d * (r1 + r3) + r2 + r4) <= mu]
            constraints += [(r1 - r3 - q / d + d * (r2 - r4) - d * (1 - q)) >= 0]
            constraints += [(r1 - r3 - q / d) >= 0]
            prob = cp.Problem(cp.Minimize(mu), constraints)
            prob.solve()
            c.append(log2(prob.value))

        # plt.plot(q_vals, c)
        if (d <= 40) or d == 100:
            plt.plot(t, c, label="d = " + str(d))
            plt.legend(loc="upper right")
        else:
            plt.plot(t, c)

        plt.title('Depolarizing Channel with d from ' + str("10") + ' to ' + str(upper))
        # plt.xlabel('q-parameter')
        plt.xlabel('t-parameter')
        plt.ylabel("log" + r'$\Gamma(N)$')
        plt.savefig('./plot/depol/lp_depol_' + str("10") + '_' + str(upper) + '.png')

        # plt.show()
    end = time.time()
    print(end - start)

print(plot_all_lp_depol(100))
