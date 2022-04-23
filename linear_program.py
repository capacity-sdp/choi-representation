import numpy as np
import cvxpy as cp
from numpy import log2
import matplotlib.pyplot as plt
import time
import choi_plot as cplot


def phi(dimension):
    d = dimension
    arr = np.zeros((d,d))



# plots the capacity bound for depolarizing/ werner channel with a given d
def plot_(dimension, channel):
    start = time.time()
    d = dimension
    D = d ** 2
    c = []

    q_vals = []
    if channel[0] == "d":
        q_vals = np.arange(0, d ** 2 / (d ** 2 - 1), 0.01, dtype=float)
    if channel[0] == "w":
        q_vals = np.arange(d / (d + 1), d / (d - 1), 0.01, dtype=float)

    r1 = cp.Variable(1)
    r2 = cp.Variable(1)
    r3 = cp.Variable(1)
    r4 = cp.Variable(1)
    iMat = np.identity(d ** 2)
    mu = cp.Variable()

    for q in q_vals:
        choi = cplot.choi_(q, d, channel)
        constraints = [Vab >> 0]
        constraints += [Yab >> 0]
        constraints += [cp.partial_transpose(Vab - Yab, (d, d), 1) >> choi]
        constraints += [Va + Ya << mu * iMat]
        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve()
        c.append(log2(prob.value))

    plt.plot(q_vals, c, label="d = " + str(d))
    plt.legend(loc="upper left")

    if channel[0] == "d":
        plt.title('Depolarizing Channel with d=' + str(d))
        plt.xlabel('q-parameter')
        plt.ylabel('quantum capacity')
        plt.savefig('./plot/depol/depol_' + str(d) + '.png')

    if channel[0] == "w":
        plt.title('Werner-Holevo Channel with d=' + str(d))
        plt.xlabel('p-parameter')
        plt.ylabel("log" + r'$\Gamma(N)$' + " - quantum capacity upperbound")
        plt.savefig('./plot/werner/wern_' + str(d) + '.png')

    # plt.show()
    end = time.time()
    print(end - start)


# # plot a general plot of capacity bound of depolar/werner channels with a given lower/upper d
# def plot_all_(lower, upper, channel):
#     start = time.time()
#     for i in range(lower, upper + 1):
#         d = i
#         D = d ** 2
#         c = []
#
#         q_vals = []
#         t = np.arange(0, 1, 0.01, dtype=float)
#         if channel[0] == "d":
#             # q_vals = np.arange(0, d ** 2 / (d ** 2 - 1), 0.01, dtype=float)
#             q_vals = t * (d ** 2 / (d ** 2 - 1))
#         if channel[0] == "w":
#             # t = np.arange(0, 1, 0.01, dtype=float)
#             q_vals = t * (d / (d + 1)) + (1 - t) * (d / (d - 1))
#             # q_vals = d * ((1 - t) / (d + 1) + t / (d - 1))
#             # print(q_vals)
#             # q_vals = np.arange(d / (d + 1), d / (d - 1), 0.01, dtype=float)
#
#         Vab = cp.Variable((D, D), symmetric=True)
#         Yab = cp.Variable((D, D), symmetric=True)
#         mu = cp.Variable()
#         iMat = np.identity(d)
#
#         Va = cp.partial_trace(Vab, (d, d), 1)
#         Ya = cp.partial_trace(Yab, (d, d), 1)
#
#         for q in q_vals:
#             choi = choi_(q, d, channel)
#             constraints = [Vab >> 0]
#             constraints += [Yab >> 0]
#             constraints += [cp.partial_transpose(Vab - Yab, (d, d), 1) >> choi]
#             constraints += [Va + Ya << mu * iMat]
#             prob = cp.Problem(cp.Minimize(mu), constraints)
#             prob.solve()
#             c.append(log2(prob.value))
#
#         # plt.plot(q_vals, c)
#         if (6 >= d >= 2) or d == 10:
#             plt.plot(t, c, label="d = " + str(d))
#             plt.legend(loc="upper right")
#         else:
#             plt.plot(t, c)
#
#     if channel[0] == "d":
#         plt.title('Depolarizing Channel with d from ' + str(lower) + ' to ' + str(upper))
#         # plt.xlabel('q-parameter')
#         plt.xlabel('t-parameter')
#         plt.ylabel("log" + r'$\Gamma(N)$')
#         # plt.savefig('./plot/depol/depol_' + str(lower) + '_' + str(upper) + '.png')
#         plt.ylim([-0.05, 1])
#         # plt.margins(x=0.1)
#         plt.savefig('./plot/depol/depol_' + str(lower) + '_' + str(upper) + '_bound.png')
#
#     if channel[0] == "w":
#         plt.title('Werner-Holevo Channel with d from ' + str(lower) + ' to ' + str(upper))
#         # plt.xlabel('p-parameter')
#         plt.xlabel('t-parameter')
#         plt.ylabel("log" + r'$\Gamma(N)$')
#         plt.savefig('./plot/werner/wern_' + str(lower) + '_' + str(upper) + '.png')
#
#         # plt.show()
#     end = time.time()
#     print(end - start)
