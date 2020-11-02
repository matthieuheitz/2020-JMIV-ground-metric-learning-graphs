#!/usr/bin/env python
"""
ml_core.py

Core functions to do Metric Learning with auto diff by Pytorch (autograd):

"""


import time

import torch
import torch.sparse

from ml_parameters import *
import np_utils as npu
import ml_kernels as mlk

__author__ = "Matthieu Heitz"


# 2 functions to debug intermediate values of gradient.
# Found here: https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/22
# Use as follow: for an intermediate variable A : mlc.register_hook(A, "grad_A") "grad_A" is just a display name
def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        print("")
        if tensor.nelement() == 1:
            # print(f"{msg} {tensor}")
            print(msg,tensor)
        else:
            # print(f"{msg} shape: {tensor.shape}"
            #       f" max: {tensor.max()} min: {tensor.min()}"
            #       f" mean: {tensor.mean()}")
            print(msg,"shape:",tensor.shape,
                  "max:",tensor.max(),"min:",tensor.min(),
                  "mean:",tensor.mean())
            print("Stats: ",stats_on_torch_tensor(tensor))
            print("Stats nz: ",stats_on_torch_tensor_nz(tensor))
            print(tensor)
    return printer

def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def stats_on_torch_tensor_nz(x):
    return npu.stats_on_array_nz(x.detach().cpu().numpy())


def stats_on_torch_tensor(x):
    return npu.stats_on_array(x.detach().cpu().numpy())


def sinkhorn_barycenter_torch(P, w, xi, L, epsilon=0, leaf=0, return_on_nan=False):
    """
    Computation of the barycenter of all flattened histograms in P, with weigths w.
    Computation is done with PyTorch containers

    Parameters
    ----------
    P : torch.Tensor of shape [K,N]
        Input histograms flattened
    w : torch.Tensor of shape [K]
        Weights
    xi : lambda function
        Kernel operator (exp(-cost/gamma))
    L : int
        Number of Sinkhorn iterations
    leaf : torch.Tensor
        Leaf variable to check the gradient. Just for debug purposes

    Returns
    -------
    q : torch.Tensor of shape [N]
        Wasserstein barycenter
    err_p : np.array([L])
        Evolution of the constraint error on the input histograms
    err_q : np.array([L])
        Evolution of the constraint error on the barycenter

    """

    K = P.shape[0]  # Number of histograms
    N = P.shape[1]  # Histogram size

    # Special case if one weight is 1 and all others are 0
    # This actually gains a general factor of 2 for N=50x50, because the gradient code
    # is super slow when differentiating this code with binary weights.
    # It also reduces memory usage.
    if torch.any(w == 1):
        # Find which one it is
        ind = torch.where(w == 1)[0].item()
        return xi(P[ind] / xi(torch.ones(N))), np.zeros(L), np.zeros(L)

    a = prm.from_numpy(np.zeros([K, N]))  # Initialize it to be able to compute err_p the first time
    b = prm.from_numpy(np.ones([K, N]))
    q = prm.from_numpy(np.zeros(N))

    err_p = np.zeros(L)
    err_q = np.zeros(L)

    for i in range(L):

        # print(i)
        list_a = []
        for k in range(K):
            xib = xi(b[k])
            err_p[i] = err_p[i] + (torch.norm(a[k] * xib - P[k]) / torch.norm(P[k])).item()
            list_a.append(P[k] / xib)
        a = torch.stack(list_a)

        if err_p[i] < epsilon:
            print("Break at iter", i)
            break

        list_xia = []
        for k in range(K):
            list_xia.append(xi(a[k]))
        xia = torch.stack(list_xia)

        q = 0
        for k in range(K):
            q = q + w[k] * torch.log(xia[k])  # Bonneel et al : p = \prod(K^Ta)^w_s
            # q = q + w[k] * torch.log(b[k]*xi(a[k])) # Benamou et al : p = \prod(b*K^Ta)^w_s
        q = torch.exp(q)

        list_b = []
        for k in range(K):
            err_q[i] = err_q[i] + (torch.norm(b[k] * xia[k] - q) / torch.norm(q)).item()
            list_b.append(q / xia[k])
        b = torch.stack(list_b)

        if not np.isfinite(torch.sum(q).item()):
            print("Reached numerical limit in Sinkhorn barycenter at iter %d." % i)
            if return_on_nan:
                break
            else:
                exit(-1)

    return q, err_p, err_q


def sinkhorn_displ_interp_wass_prog(P, S, xi, L, epsilon=0, leaf=0, output_known_labels=False):
    """
    Simplify the call to wasserstein_propagation()
    for the case of a displacement interpolation between 2 probability vectors.

    :param P: Two probability vectors that are the extremities of the interpolation
    :param S: Number of points in the interpolation (including extremities)
    :param xi: lambda function for the kernel operator
    :param L: number of iterations
    :param epsilon: stopping threshold
    :param leaf: leaf variable to check gradient
    :param output_known_labels: include blurred extremities
    :return: Array of interpolated probability vectors and error vectors
    """

    V = np.arange(S)
    V0 = {}
    V0[0] = P[0]
    V0[S-1] = P[1]
    E = np.arange(S).repeat(2)[1:-1].reshape(-1,2)
    w = np.ones(S-1)
    return wasserstein_propagation(V,V0,E,w,xi,L,epsilon,leaf,output_known_labels)


def wasserstein_propagation(V, V0, E, w, xi, L, epsilon=0, leaf=0, output_known_labels=False):
    """
    Implement the Wasserstein Propagation algorithm in Solomon et al. [2014] : Convolutional Wasserstein Distances

    :param V: array of labels defining the vertices of the graph
    :param V0: dictionary that gives a probability vector only for the "known" labels
    :param E: array of edges (pairs of labels)
    :param w: array of weights for each edge
    :param xi: lambda function for the kernel operator
    :param L: number of iterations
    :param epsilon: stopping threshold
    :param leaf: leaf variable to check gradient
    :param output_known_labels: whether to include blurred known labels in the output
    :return: Array of propagated probability vectors, and error vectors
    """

    A = E.shape[0] # Number of edges
    N = V0[next(iter(V0))].shape[0] # Get the size of the histograms (take the first key with an iterator)
    v_e = prm.from_numpy(np.ones([A,N]))      # Scaling vectors
    w_e = prm.from_numpy(np.ones([A,N]))      # Scaling vectors
    # TODO : Maybe use a list instead ? To declare a list of given size : l = [None] * 10
    # The access in a dict is log(N), and is constant in a list.
    # If N is small, it doesn't change much...
    # Nv = len(V)
    # mu_v = [None] * Nv   # Vector of probability vectors to compute
    mu_v = {}   # Dict of probability vectors to compute

    # Add blurred known distributions. It's divided by a blurred ones vector that keeps it normalized
    if output_known_labels:
        for v in V0:
            mu_v[v] = xi(V0[v]/xi(prm.tensor(torch.ones(N))))

    # Begin Sinkhorn iterations
    for i in range(L):
        for v in V:     # For each vertex
            if v in V0:     # if v is known
                mu = V0[v]      # Get the probability vector of the vertex
                for e in range(A):  # Index over all edges
                    if v in E[e]:  # If edge adjacent to v (neighbors of v)
                        if v == E[e, 1]: w_e[e] = mu / xi(v_e[e])
                        if v == E[e, 0]: v_e[e] = mu / xi(w_e[e])
            else:           # if v is an unknown vertex
                w_tot = 0   # Initialize total weight
                N_v = []    # Initialize list of neighbor vertices
                d_e = {}    # Initialize intermediate variable
                mu_v[v] = prm.from_numpy(np.ones(N)) # Initialize interpolated probability vector
                for e in range(A):  # Index over all edges
                    if v in E[e]:  # If edge adjacent to v (neighbors of v)
                        N_v.append(e)
                        w_tot += w[e]
                        d_e[e] = prm.from_numpy(np.zeros(N))
                for e in N_v:
                    # if v == E[e, 1]: d_e[e] = w_e[e] * xi(v_e[e])   # Not necessary because the product of the potentials is 1.
                    # if v == E[e, 0]: d_e[e] = v_e[e] * xi(w_e[e])   # Not necessary because the product of the potentials is 1.
                    if v == E[e, 1]: d_e[e] = xi(v_e[e])
                    if v == E[e, 0]: d_e[e] = xi(w_e[e])
                    mu_v[v] = mu_v[v] * d_e[e]**(w[e]/w_tot)     # Compute the barycenter between neighbors

                for e in N_v:
                    # if v == E[e, 1]: w_e[e] = w_e[e] * mu_v[v] / d_e[e]   # Not necessary to remultiply by w_e
                    # if v == E[e, 0]: v_e[e] = v_e[e] * mu_v[v] / d_e[e]   # Not necessary to remultiply by v_e
                    if v == E[e, 1]: w_e[e] = mu_v[v] / d_e[e]
                    if v == E[e, 0]: v_e[e] = mu_v[v] / d_e[e]

                # # Export histograms
                # n = int(round(np.sqrt(N)))
                # I = prm.tensor2npy(mu_v[v]).reshape(n,n)
                # imageio.imsave(os.path.join(prm.outdir, "imagewp-%02d-%02d-%02d.png" % (prm.iter_num,i,v)), I/np.max(I))
                # np.save(os.path.join(prm.outdir, "arraywp-%02d-%02d-%02d" % (prm.iter_num, i, v)), I)

    # TODO : Return also some error vectors to check for convergence
    return torch.stack([mu_v[v] for v in sorted(mu_v)])




def compute_kernel_torch(A, n, dim, apsp_algorithm, extra_param):
    """
    Compute the kernel operator, and the kernel matrix when possible, with the given apsp algorithm.

    :param A: an array of weights, corresponding to weights on points, or edges
    :param n:
    :param dim:
    :param apsp_algorithm:
    :param extra_param: dictionary of extra parameters.
    :return:
    """

    t0 = time.time()
    xi = lambda x : x
    exp_C = prm.tensor([0])
    N = n**dim
    h = 1/(n-1) # Grid resolution

    # Get metric values on edges instead of points
    if extra_param['metric_type'].startswith("grid_vertices"):
        # This is not an in-place operation
        A = get_metric_values_from_points_to_edges_torch(A, n, dim)

    # From here on, the weights in A are weights on the graph's edges.
    if apsp_algorithm == "Numpy_kernel":

        # Get parameters
        t_final = extra_param['t_heat']
        K = extra_param['k_heat']
        alpha = extra_param['alpha']

        if prm.solver_type.startswith("Sparse"):

            if (not 'kernel_version' in extra_param) or (extra_param['kernel_version'] == "kernel3"):

                mlk.NumpyKernelFunctionNew3.precomputation(A, extra_param)

                # Do substepping inside the kernel
                def xi(x):
                    return mlk.NumpyKernelFunctionNew3.apply(A, x, extra_param)

            else:
                print("Unrecognized value for 'kernel_version'")
                exit(-1)

        else:
            raise ValueError("Solver type '%s' not recognized"%prm.solver_type)

    else:
        print("Error: APSP algorithm name : '%s' unrecognized."%apsp_algorithm)

    # print("Kernel Setup: %f" % (time.time() - t0))

    return xi, exp_C, extra_param



def compute_2d_euclidean_kernel_native(n, gamma):

    import ctypes as ct

    N = n**2
    dtype = 'float64'
    ctypes_dtype = ct.c_double
    conv_method = "c_omp"   # c_omp, cpp_omp, cpp_halide

    # if conv_method == "c":
    # Load the shared lib
    cdll_name = "libckernels.so"
    # cdllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + cdll_name
    cdllabspath = os.path.join(os.path.dirname(os.path.abspath(__file__)),cdll_name)
    myclib = ct.CDLL(cdllabspath)
    # Define argument types
    myclib.convolution_batch_2d.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ct.c_int, ct.c_int,ct.c_int]
    # myclib.convolution_batch_build_kernel.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ctypes_dtype, ct.c_int, ct.c_int, ct.c_int,ct.c_int]
    # myclib.my_conv_batch.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ctypes_dtype, ct.c_int, ct.c_int, ct.c_int,ct.c_int]

    # Build 1d kernel
    t = np.linspace(0, 1, n).astype(dtype)
    k1d = np.exp(-t**2 / gamma)

    def xi(x):
        nv = 1
        transposed = False
        if x.ndim == 2:
            # Puts the batch dimension on the lines so that flatten() puts them one after the other.
            if x.shape[0] > x.shape[1]:
                transposed = True
                x = x.T
            nv = x.shape[0]
            if x.shape[1] != N:
                print("Error: vectors are not of size", N)
                exit(-1)
        elif x.ndim == 1:
            if x.shape[0] != N:
                print("Error: vector is not of size", N)
                exit(-1)
        else:
            print("compute_2d_euclidean_kernel_native.xi(): Only arrays of dimension 1 or 2 are supported.")
            exit(-1)
        if not x.flags['C']: # If array is not contiguous
            print("WARNING: Input array is not contiguous !")
            x = np.ascontiguousarray(x)

        # USING C LIB
        res = np.zeros([nv, N]).astype(dtype).squeeze()
        x_p = x.ctypes.data_as(ct.POINTER(ctypes_dtype))
        res_p = res.ctypes.data_as(ct.POINTER(ctypes_dtype))
        k1d_p = k1d.ctypes.data_as(ct.POINTER(ctypes_dtype))    # This should be kept inside of xi() otherwise k1d will be destroyed earlier than expected
        myclib.convolution_batch_2d(x_p, k1d_p, res_p, n, n, nv)

        if transposed: res = res.T
        return res.squeeze()

    return xi



def compute_3d_euclidean_kernel_native(n, gamma):

    import ctypes as ct
    import os

    N = n**3
    dtype = 'float64'
    ctypes_dtype = ct.c_double
    conv_method = "c_omp"   # c_omp, cpp_omp, cpp_halide

    # Load the shared lib
    cdll_name = "libckernels.so"
    cdllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + cdll_name
    myclib = ct.CDLL(cdllabspath)
    # Define argument types
    myclib.convolution_batch.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ct.c_int, ct.c_int, ct.c_int,ct.c_int]
    myclib.convolution_batch_build_kernel.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ctypes_dtype, ct.c_int, ct.c_int, ct.c_int,ct.c_int]
    myclib.my_conv_batch.argtypes = [ct.POINTER(ctypes_dtype), ct.POINTER(ctypes_dtype), ctypes_dtype, ct.c_int, ct.c_int, ct.c_int,ct.c_int]

    # Build 1d kernel
    t = np.arange(n).astype(dtype)/(n-1)
    k1d = np.exp(-t**2 / gamma)

    def xi(x):
        nv = 1
        transposed = False
        if x.ndim == 2:
            # Puts the batch dimension on the lines so that flatten() puts them one after the other.
            if x.shape[0] > x.shape[1]:
                transposed = True
                x = x.T
            nv = x.shape[0]
            if x.shape[1] != N:
                print("Error: vectors are not of size", N)
                exit(-1)
        elif x.ndim == 1:
            if x.shape[0] != N:
                print("Error: vector is not of size", N)
                exit(-1)
        else:
            print("compute_3d_euclidean_kernel_native.xi(): Only arrays of dimension 1 or 2 are supported.")
            exit(-1)
        if not x.flags['C']: # If array is not contiguous
            print("WARNING: Input array is not contiguous !")
            x = np.ascontiguousarray(x)

        # USING C LIB
        res = np.zeros([nv, N]).astype(dtype).squeeze()
        x_p = x.ctypes.data_as(ct.POINTER(ctypes_dtype))
        res_p = res.ctypes.data_as(ct.POINTER(ctypes_dtype))
        k1d_p = k1d.ctypes.data_as(ct.POINTER(ctypes_dtype))    # This should be kept inside of xi() otherwise k1d will be destroyed earlier than expected
        myclib.convolution_batch(x_p, k1d_p, res_p, n, n, n, nv)
        # I think the reason that k1d is destroyed if we put the "k1d_p = k1d.ctypes.data_as(...)" line outside of xi,
        # is that there is no more reference of k1d in xi, so it is destroyed after some point.
        # Putting this line here probably keeps the reference counter of k1d to 1, because xi has a reference on it,
        # so it is not destroyed.

        if transposed: res = res.T
        return res.squeeze()

    return xi



def l2norm_of_laplacian_torch(w,n,dim,extra_prm):

    if w.shape != (dim,n**dim):
        print("Error: w should be of shape (dim, n**dim)")
        exit(-1)

    metric_on_grid_vertices = (extra_prm['metric_type'] == "grid_vertices_tensor_diag")
    z = 0

    if dim == 1:

        if metric_on_grid_vertices:
            B = prm.from_numpy(np.zeros(n))
            W0 = w.clone()
        else:   # Remove padded zeros
            B = prm.from_numpy(np.zeros(n - 1))
            W0 = w.clone()[:, :-1]

        B[0] += 1
        B[-1] += 1
        B = 2*dim - B

        L0 = B * W0
        L0[:, :-1] -= W0[:, 1:]
        L0[:, 1:] -= W0[:, :-1]

        z = torch.norm(L0)

    if dim == 2:

        if metric_on_grid_vertices:
            W0 = w.clone()[0].reshape([n,]*dim)
            W1 = w.clone()[1].reshape([n,]*dim)
            B = prm.from_numpy(np.zeros([n,]*dim))
        else:   # Remove padded zeros
            W0 = w.clone()[0].reshape([n,]*dim)[:, :-1]
            W1 = w.clone()[1].reshape([n,]*dim)[:-1, :]
            B = prm.from_numpy(np.zeros([n,n-1]))

        # Create degree matrix (number of neighbors for each pixel)
        B[ 0,:] += 1
        B[-1,:] += 1
        B[:, 0] += 1
        B[:,-1] += 1
        B = 2*dim - B
        # Compute laplacian of the metric (dimension 0)
        L0 = B * W0
        L0[:-1] -= W0[1:]
        L0[1:] -= W0[:-1]
        L0[:, :-1] -= W0[:, 1:]
        L0[:, 1:] -= W0[:, :-1]
        # Compute laplacian of the metric (dimension 1)
        L1 = B.t() * W1     # Works for both metric_on_grid_vertices=True or False
        L1[:-1] -= W1[1:]
        L1[1:] -= W1[:-1]
        L1[:, :-1] -= W1[:, 1:]
        L1[:, 1:] -= W1[:, :-1]

        z = torch.norm(torch.cat((L0, L1.t())))

    if dim == 3:

        if metric_on_grid_vertices:
            W0 = w.clone()[0].reshape([n,]*dim)
            W1 = w.clone()[1].reshape([n,]*dim)
            W2 = w.clone()[2].reshape([n,]*dim)
            B = prm.from_numpy(np.zeros([n,]*dim))
        else:   # Remove padded zeros
            W0 = w.clone()[0].reshape([n,]*dim)[:, :, :-1]
            W1 = w.clone()[1].reshape([n,]*dim)[:, :-1, :]
            W2 = w.clone()[2].reshape([n,]*dim)[:-1, :, :]
            B = prm.from_numpy(np.zeros([n,n,n-1]))

        # Create degree matrix (number of neighbors for each pixel)
        B[ 0,:,:] += 1
        B[-1,:,:] += 1
        B[:, 0,:] += 1
        B[:,-1,:] += 1
        B[:,:, 0] += 1
        B[:,:,-1] += 1
        B = 2*dim - B
        # Compute laplacian of the metric (dimension 0)
        L0 = B * W0
        L0[:-1] -= W0[1:]
        L0[1:] -= W0[:-1]
        L0[:, :-1] -= W0[:, 1:]
        L0[:, 1:] -= W0[:, :-1]
        L0[:, :, :-1] -= W0[:, :, 1:]
        L0[:, :, 1:] -= W0[:, :, :-1]
        # Compute laplacian of the metric (dimension 1)
        L1 = B.transpose(2,1) * W1      # Works for both metric_on_grid_vertices=True or False
        L1[:-1] -= W1[1:]
        L1[1:] -= W1[:-1]
        L1[:, :-1] -= W1[:, 1:]
        L1[:, 1:] -= W1[:, :-1]
        L1[:, :, :-1] -= W1[:, :, 1:]
        L1[:, :, 1:] -= W1[:, :, :-1]
        # Compute laplacian of the metric (dimension 2)
        L2 = B.transpose(2,0) * W2      # Works for both metric_on_grid_vertices=True or False
        L2[:-1] -= W2[1:]
        L2[1:] -= W2[:-1]
        L2[:, :-1] -= W2[:, 1:]
        L2[:, 1:] -= W2[:, :-1]
        L2[:, :, :-1] -= W2[:, :, 1:]
        L2[:, :, 1:] -= W2[:, :, :-1]

        z = torch.norm(torch.cat((L0.transpose(2, 1), L1, L2.transpose(0, 1))))

    return z



def get_metric_values_from_points_to_edges_torch(w,n,dim):
    # The output weights are zero-padded.
    N = n**dim
    W = None
    if dim == 1:
        W0 = w.clone()[0]
        W = prm.from_numpy(np.zeros([dim]+[n,]*dim))
        W[0,:-1] = ((W0[:-1]+W0[1:])/2)
        W = W.reshape(dim,N)
    if dim == 2:
        W0 = w.clone()[0].reshape([n,]*dim)
        W1 = w.clone()[1].reshape([n,]*dim)
        W = prm.from_numpy(np.zeros([dim]+[n,]*dim))
        W[0,:,:-1] = ((W0[:,:-1]+W0[:,1:])/2)
        W[1,:-1,:] = ((W1[:-1]+W1[1:])/2)
        W = W.reshape(dim,N)
    if dim == 3:
        W0 = w.clone()[0].reshape([n,]*dim)
        W1 = w.clone()[1].reshape([n,]*dim)
        W2 = w.clone()[2].reshape([n,]*dim)
        W = prm.from_numpy(np.zeros([dim]+[n,]*dim))
        W[0,:,:,:-1] = ((W0[:,:,:-1]+W0[:,:,1:])/2)
        W[1,:,:-1,:] = ((W1[:,:-1]+W1[:,1:])/2)
        W[2,:-1,:,:] = ((W2[:-1]+W2[1:])/2)
        W = W.reshape(dim,N)
    return W


def loss_TV(p,q):
    """
    Total Variation loss function between two torch Tensors

    :param p: 1st tensor
    :param q: 2nd tensor
    :return: sum of the absolute differences of each component
    """
    return torch.sum(torch.abs(p-q))    # Already size-independent


def loss_L2(p,q):
    """
    L2 loss function between two torch Tensors

    :param p: 1st tensor
    :param q: 2nd tensor
    :return: sum of the squared differences of each component
    """
    return torch.dot(p-q,p-q)*p.numel()     # Multiply by numel to make it size-independent


def loss_KL(p,q):
    """
    Kulback-Leibler divergence between two torch Tensors

    :param p: 1st tensor
    :param q: 2nd tensor
    :return: Kullback-Leibler divergence of p and q
    """
    return torch.sum(p*torch.log(p/q)-p+q)  # Already size-independent


def loss_W(p, q, xi, L, gamma, j=0):
    """
    Wasserstein distance between two torch Tensors

    :param p: 1st tensor
    :param q: 2nd tensor
    :param xi: cost kernel
    :param L: Number of iterations
    :param gamma: regularization parameter
    :return: Wasserstein distance between p and q
    """
    a = torch.ones_like(p)
    b = a.clone()
    err_p = np.zeros([L])
    err_q = np.zeros([L])

    for i in range(0,L):
        xib = xi(b)
        with torch.no_grad():
            err_p[i] = (torch.norm(a* xib - p) / torch.norm(p)).item()
        a = p / xib
        xia = xi(a)
        with torch.no_grad():
            err_q[i] = (torch.norm(b* xia - q) / torch.norm(q)).item()
        b = q / xia

    # Sinkhorn distance (formula in loss.h, verified)
    return gamma*(torch.dot(p,torch.log(a)) + torch.dot(q,torch.log(b)) - torch.dot(a,xi(b))), err_p, err_q


def loss_W_normalized(p, q, xi, L, gamma, j):

    Wpq, err_pq_p, err_pq_q = loss_W(p, q, xi, L, gamma)
    Wpp, err_pp_p, err_pp_q = loss_W(p, p, xi, L, gamma)
    Wqq, err_qq_p, err_qq_q = loss_W(q, q, xi, L, gamma)

    return 2*Wpq - Wpp - Wqq, err_pq_p, err_pq_q

