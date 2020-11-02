#!/usr/bin/env python
"""
np_utils.py

Utility functions for Metric Learning
N.B : None of these functions require PyTorch

"""

import numpy as np
from scipy import sparse
import json


__author__ = "Matthieu Heitz"


def write_dict_to_json(fp,dictionary,append=False,indent_level=0):

    new_str = json.dumps(dictionary, indent=indent_level, sort_keys=True)

    # For replacing in bigger files, use fileinput
    # See https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file-using-python/20593644#20593644
    if append:
        # Read in the file
        with open(fp, 'r') as file:
            filedata = file.read()

        # Replace the target string
        offset = 1 if indent_level == None else 2   # indent=None : remove "{", otherwise remove "{\n",
        filedata = filedata.replace('\n}', ",\n" + new_str[offset:])

        # Write the file out again
        with open(fp, 'w') as file:
            file.write(filedata)
    else:
        with open(fp, 'w') as file:
            file.write(new_str)


def stats_on_array_nz(x):
    y = x[np.nonzero(x)]
    return 'nz [min,mean,max]:[%e,%e,%e]'%(np.min(y),np.mean(y),np.max(y))


def stats_on_array(x):
    return '[min,mean,max]:[%e,%e,%e]'%(np.min(x),np.mean(x),np.max(x))


def peaks(x, y):
    return 3 * (1 - x) ** 2. * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)


def peaks_matlab(n=49):

    t = np.linspace(-3,3,n)
    x, y = np.meshgrid(t, t)
    z = peaks(x, y)
    from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(x, y, z)
    return z


def gaussian1d(x,mu,sigma):
    return np.exp(-(x - mu) ** 2 / (2 * (sigma) ** 2))


def gaussian2d(x,y,mu,sigma):
    return np.exp((-(x - mu[0]) ** 2 - (y - mu[1]) ** 2) / (2 * (sigma) ** 2))


def gaussian3d(x,y,z,mu,sigma):
    return np.exp((-(x - mu[0]) ** 2 - (y - mu[1]) ** 2 - (z - mu[2]) ** 2)/ (2 * (sigma) ** 2))


def multidim_gaussian(x,y,mu,S):
    from scipy.stats import multivariate_normal
    pos = np.stack((y, x)).T
    rv = multivariate_normal(mu, S)
    return rv.pdf(pos)


def multidim_gaussian_angle(x,y,mu,lx,ly,theta):
    import math
    cos2 = math.cos(theta) * math.cos(theta)
    sin2 = math.sin(theta) * math.sin(theta)
    sincos = math.sin(theta) * math.cos(theta)
    M = np.array([[lx * cos2 + ly * sin2, (lx - ly) * sincos],
                  [(lx - ly) * sincos, lx * sin2 + ly * cos2]])
    return multidim_gaussian(x,y,mu,M)


def get_grid_weighted_adjacency_npsparse(n, dim, W):
    if dim == 1:
        return get_1dgrid_weighted_adjacency_npsparse(n, W)
    elif dim == 2:
        return get_2dgrid_weighted_adjacency_npsparse(n, W)
    elif dim == 3:
        return get_3dgrid_weighted_adjacency_npsparse(n, W)
    else:
        print("ERROR: Function not implemented for dim > 3")
        return -1


def get_1dgrid_weighted_adjacency_npsparse(n, W):
    """
    Computes the adjacency matrix (2-neighborhood) of pixels in
    a 1D grid of size n, using the weight vector passed.
    Stores it in a sparse diagonal matrix

    :param n: Segment size
    :type n: int
    :param w: weights of adjacencies
    :type w: np.ndarray([1,n-1])
    :return: The adjacency matrix
    :rtype: scipy.sparse.dia_matrix
    """
    D = W.squeeze() # Drop the extra dimension if there is one
    if W.shape[1] == n-1:
        D = insert_zeros_in_1d_weight_vector(n, W)
    elif W.shape[1] != n:
        raise ValueError("Incorrect size of weight vectors: should be n or n-1.")

    D1 = D
    A = sparse.dia_matrix((np.stack((D1,np.roll(D1,1)),axis=0), [-1,1]), shape=(n,n))
    return A


def get_2dgrid_weighted_adjacency_npsparse(n, W):
    """
    Computes the adjacency matrix (4-neighborhood) of pixels in
    a square 2D grid of size (n,n), using the horizontal
    and vertical weight vectors passed.
    Stores it in a sparse diagonal matrix

    :param n: Image size
    :type n: int
    :param W: Weights of fast axis (first row, horizontal, columns) and slow axis (second row, vertical, rows) adjacencies
    :type W: np.ndarray([2,n*(n-1)])
    :return: The adjacency matrix
    :rtype: scipy.sparse.dia_matrix
    """
    if W.shape[0] == 2:
        D = W
        if W.shape[1] == n*(n-1):
            D = insert_zeros_in_2d_weight_vector(n, W)
        elif W.shape[1] != n**2:
            raise ValueError("Incorrect size of weight vectors: should be n**2 or n*(n-1).")

        D1 = D[0]; Dn = D[1]
        data = np.stack((Dn, np.roll(Dn,n), D1, np.roll(D1,1)), axis=0)
        offsets = np.array([-n, n, -1, 1])
    elif W.shape[0] == 4:
        D = W
        if W.shape[1] != n ** 2:
            raise ValueError("Incorrect size of weight vectors: should be n**2.")
        D1 = D[0]
        Dn = D[1]
        Dnp1 = D[2]
        Dnm1 = D[3]
        data = np.stack((Dnp1, Dn, Dnm1, D1, np.roll(D1, 1), np.roll(Dnm1, n - 1), np.roll(Dn, n), np.roll(Dnp1, n + 1)), axis=0)
        offsets = np.array([-n - 1, -n, -n + 1, -1, 1, n - 1, n, n + 1])

    A = sparse.dia_matrix((data, offsets), shape=(n**2, n**2))
    return A


def get_3dgrid_weighted_adjacency_npsparse(n, W):
    """
    Computes the adjacency matrix (8-neighborhood) of pixels in
    a square 3D grid of size (n,n,n), using the weight vector passed.
    Stores it in a sparse diagonal matrix

    :param n: Image size
    :type n: int
    :param W: Weights of fast axis (first row, color B, z), medium axis (second row, color G, y), and slow axis (third row, color R, x) adjacencies
    :type W: np.ndarray([3,n**2*(n-1)])
    :return: The adjacency matrix
    :rtype: scipy.sparse.dia_matrix
    """
    D = W
    if W.shape[1] == n**2*(n-1):
        D = insert_zeros_in_3d_weight_vector(n, W)
    elif W.shape[1] != n**3:
        raise ValueError("Incorrect size of weight vectors: should be n**3 or n**2*(n-1).")

    Dn2 = D[2]; Dn = D[1]; D1 = D[0]
    offsets = np.array([-n**2,n**2,-n, n, -1, 1])
    data = np.stack((Dn2, np.roll(Dn2, n**2), Dn, np.roll(Dn, n), D1, np.roll(D1, 1)), axis=0)
    A = sparse.dia_matrix((data, offsets), shape=(n**3, n**3))
    return A


def insert_zeros_in_weight_vector(n, dim, W):
    if dim == 1:
        return insert_zeros_in_1d_weight_vector(n, W)
    elif dim == 2:
        return insert_zeros_in_2d_weight_vector(n, W)
    elif dim == 3:
        return insert_zeros_in_3d_weight_vector(n, W)
    else:
        print("ERROR: Function not implemented for dim > 3")
        return -1


def insert_zeros_in_1d_weight_vector(n, W):
    if W.shape[1] != n-1:
        raise ValueError("Incorrect size of weight vectors: should be n-1.")

    return np.append(W,np.zeros(1)).reshape([1,-1])


def insert_zeros_in_2d_weight_vector(n, W):
    if W.shape[1] != n*(n-1):
        raise ValueError("Incorrect size of weight vectors: should be n*(n-1).")

    Dn0 = np.insert(W[0,:],np.arange(n-1,n*(n-1)+1,n-1),np.zeros(n),axis=0)
    Dn1 = np.append(np.array([W[1,:]]),np.zeros(n))  # Must pad zeros for upper diagonals
    return np.stack((Dn0,Dn1))


def insert_zeros_in_3d_weight_vector(n, W):
    if W.shape[1] != n**2*(n-1):
        raise ValueError("Incorrect size of weight vectors: should be n**2*(n-1).")

    Dn0 = np.insert(W[2,:], np.arange(n - 1, n**2*(n-1)+1, n - 1), np.zeros(n ** 2), axis=0)
    Dn1 = np.insert(W[1,:], np.repeat(np.arange(n*(n-1), n**2*(n-1)+1,n*(n-1)),n), np.zeros(n**2))
    Dn2 = np.append(np.array([W[0, :]]), np.zeros(n**2))  # Must pad zeros for upper diagonals
    return np.stack((Dn0,Dn1,Dn2))


# Removes the zeros inserted by the previous functions, not all zeros.
# If there are zeros where the mask is 1, they will stay in the vector.
def remove_zeros_in_weight_vector(n, W, dim):
    mask = insert_zeros_in_weight_vector(n, dim, np.ones([dim, n ** (dim - 1) * (n - 1)])).astype('bool')
    return W.flatten()[mask.flatten()].reshape([dim, n**(dim-1) * (n-1)]).squeeze()
