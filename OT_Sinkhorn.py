#!/usr/bin/env python
"""
OT-Sinkhorn.py:

Computes Optimal Transport between 2 distributions with the Sinkhorn algorithm
Pytorch is not used at all, only numpy and scipy.
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

__author__ = "Matthieu Heitz"


def pdf_entropy(p):
    return -np.sum(p*np.log(p))


def compute_sinkhorn_barycenter(P, w, xi, L, entropic_sharpening=False):
    """
    Computation of the barycenter of all flattened histograms in P, with weigths w.
    Computation is done with Numpy containers

    :param P: Input histograms flattened
    :type P: np.array(N,K)
    :param w: Weights
    :type w: np.array(K)
    :param xi: Kernel operator (exp(-cost/gamma))
    :type xi: lambda function
    :param L: Number of Sinkhorn iterations
    :type L: int
    :return: Wasserstein barycenter
    :rtype: np.array(N)
    """

    from numpy import linalg

    K = P.shape[0] # Number of histograms
    N = P.shape[1] # Histogram size

    # Special case if one weight is 1 and all others are 0
    # This actually gains a general factor of 2 for N=50x50, because the gradient code
    # is super slow when differentiating this code with binary weights.
    # It also reduces memory usage.
    if np.any(w == 1):
        # Find which one it is
        ind = np.where(w == 1)[0]
        return xi(P[ind]/xi(np.ones(N))), np.zeros(L), np.zeros(L)

    b = np.ones([K,N])
    a = np.zeros_like(b) # Must initialize a to compute err_p the first time
    xia = np.ones([K,N])
    err_p = np.zeros(L)
    err_q = np.zeros(L)

    # Take H0 as maximum entropy of input histograms
    if entropic_sharpening:
        H0 = np.max(-np.sum(P*np.log(P), 0))

    for i in range(L):
        # print(i)
        if i % (L / 5) == 0: print("sink iter",i)
        for k in range(K):
            xib = xi(b[k]) # Pre-compute K(b)
            err_p[i] = err_p[i] + linalg.norm(a[k]*xib - P[k])/linalg.norm(P[k])
            a[k] = P[k]/xib
            xia[k] = xi(a[k]) # Pre-compute K(a)

        q = np.zeros(N)
        for k in range(K):
            q = q + w[k] * np.log(xia[k]) # Bonneel et al : p = \prod(K^Ta)^w_s
            # q = q + w[k] * np.log(b[k]*xi(a[k])) # Benamou et al : p = \prod(b*K^Ta)^w_s

        q = np.exp(q)

        # Entropic sharpening
        if entropic_sharpening:
            beta = 1
            if pdf_entropy(q) + np.sum(q) > H0 + 1:
                func = lambda x : np.sum(q**x) + pdf_entropy(q**x) - (H0 + 1)
                try:
                    beta = scipy.optimize.brentq(func,1,10)
                except (RuntimeError,ValueError):
                    print("Root not found")
                print("i = %d, beta = %g"%(i,beta))
            q = q**beta

        # Plot evolution of the barycenter
        # plt.imshow(q.reshape([70, 70]),cmap='gray')
        # plt.pause(0.1)

        for k in range(K):
            err_q[i] = err_q[i] + linalg.norm(b[k]*xia[k] - q)/linalg.norm(q) # Supposed to be K^T(a), not K(a)
            b[k] = q/xia[k]

    # plt.figure(figsize=(9,5))
    # plt.subplot(121)
    # plt.semilogy(err_p,linewidth = 2)
    # plt.title("Constraint error on input histograms")
    # plt.subplot(122)
    # plt.semilogy(err_q,linewidth = 2)
    # plt.title("Constraint error on barycenter")
    # # plt.show()

    return q, err_p, err_q

