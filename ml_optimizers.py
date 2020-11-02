#!/usr/bin/env python
"""
ml_optimizers.py

Optimizers that can be used for 2D or 3D problems.

"""

import matplotlib
# print("Matplotlib backend: ", matplotlib.get_backend())
matplotlib.use('Agg')
# print("Matplotlib backend: ", matplotlib.get_backend())
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch

from ml_parameters import *
import np_utils as npu
import ml_core as mlc


__author__ = "Matthieu Heitz"



def lbfgs_scipy(P,w,obs,loss_func,x0,L,gamma,max_iter,var_change,forward_routine,save_current_iter,extra_prm):

    assert (P.shape[1] == obs.shape[1])  # Same N
    assert (w.shape[0] == obs.shape[0])  # Same S
    N = P.shape[1]
    if prm.metric_type.startswith("grid"):
        n = extra_prm['n']
        dim = extra_prm['dim']
        if dim != x0.shape[0]:
            print("ERROR: Dimension in x0 doesn't match the one in extra_prm['dim']")
        # x0 is always of size [dim,N] when we receive it here

    if prm.metric_type == "grid_edges_scalar":
        print("\nWARNING: The padded zeros in the vector will be removed for the optimization.\n")
        x0 = npu.remove_zeros_in_weight_vector(n,x0,dim)

    # Apply change of variable before optimization
    varch_f_np = lambda x: x
    varch_b_np = lambda x: x
    if var_change:
        prm.varch_f = lambda x: torch.log(x)  # Variable change: Forward - from normal domain to other domain
        prm.varch_b = lambda x: torch.exp(x)  # Variable change: Backward - from other domain to normal domain
        varch_f_np = lambda x: np.log(x)
        varch_b_np = lambda x: np.exp(x)

    x0 = varch_f_np(x0).flatten()
    t_P = prm.from_numpy(P)
    t_w = prm.from_numpy(w)
    t_obs = prm.from_numpy(obs)
    print('Initial Point :', x0)

    if max_iter < 0: max_iter = np.inf

    # Control variables
    if extra_prm['restart_same_outdir']:
        prev_loss = np.load(os.path.join(os.path.dirname(extra_prm['init_param_file']),"1-loss.npy"))
        loss_array = prev_loss[:prm.iter_num].tolist()
    else:
        loss_array = []

    # We start at 1 because the callback is only called at iteration 1.
    prm.iter_num += 1

    cur_loss = 0
    cur_grad = 0

    plt.ioff()
    if prm.plot_loss and prm.show_plot:
        plt.ion()

    import scipy.optimize

    def numpy_func(np_x):

        print("\nNew evaluation, iter %d"%prm.iter_num)
        # Print stats on the x, as optimized by lbfgs, and stats on x in the normal domain if varchange is True.
        # print("np_x: ", np_x)
        # print("Shape np_x: ", np_x.shape)
        print("Stats np_x: ", npu.stats_on_array(np_x))
        if not var_change:
            if np.any(np_x < 0):
                print("Error: A negative value appeared in the metric")
        else:
            print("varch_b(np_x): ",npu.stats_on_array(varch_b_np(np_x)))

        if prm.metric_type.startswith("grid"):
            xp = np_x.reshape([dim, -1])
            if prm.metric_type == "grid_edges_scalar":
                xp = varch_f_np(npu.insert_zeros_in_weight_vector(n,dim,varch_b_np(xp)))
        elif prm.metric_type.startswith("graph"):
            xp = np_x
        # else:

        # Set the grad seed
        x = prm.from_numpy(xp, requires_grad=True)

        t0 = time.time()
        n_int = extra_prm['n_interp']
        loss = 0
        parallel = extra_prm['seq_parallel'] if 'seq_parallel' in extra_prm else False
        if not parallel:
            for i in range(extra_prm['n_seq']):
                extra_prm['i_seq'] = i
                i_loss = forward_routine(t_P[2*i:2*(i+1)], t_w[n_int*i:n_int*(i+1)], t_obs[n_int*i:n_int*(i+1)], loss_func, x, L, gamma, extra_prm)
                loss += i_loss
                t1 = time.time()
                print('F: %f, ' % (t1 - t0), end='', flush=True)
                i_loss.backward()   # Gradient is accumulated in x.grad
                t2 = time.time()
                print('B: %f' % (t2 - t1))
        else:
            if N*extra_prm['k_heat']*extra_prm['n_seq'] > 3e6:  # N=2500 * K=100 * n_seq=4 ~= 1e6 : 16Go. : Cuts at 3*16=48Go.
                print("ERROR: This probably won't fit in memory... \nTry setting seq_parallel to False.")
                exit(-1)
            # Use pathos to pickle local functions
            import pathos.multiprocessing
            pool = pathos.multiprocessing.Pool(extra_prm['n_seq'])    # Multi-processing (distributed memory)

            # Worker function
            def work(args):
                extra_prm['i_seq'] = args[3]
                i_loss = forward_routine(args[0], args[1], args[2], loss_func, x, L, gamma, extra_prm)
                t1 = time.time()
                print('F: %f, ' % (t1 - t0), end='', flush=True)
                i_loss.backward()   # Gradient is accumulated in x.grad
                t2 = time.time()
                print('B: %f' % (t2 - t1))
                return i_loss, x.grad

            # Build argument list
            args = []
            for i in range(extra_prm['n_seq']):
                args.append((t_P[2*i:2*(i+1)], t_w[n_int*i:n_int*(i+1)], t_obs[n_int*i:n_int*(i+1)],i))
            # Launch processes (each one copies all the data, so be careful with the memory !)
            res = pool.map(work, args)
            # Aggregate results
            x.grad = torch.zeros_like(x)
            for r in res:
                loss += r[0]
                x.grad += r[1]

        # Save grad and set it to zero (not sure it's necessary to zero it)
        grad_x = x.grad.clone()
        x.grad.zero_()

        # Save tensor variables for save_current_iter in callback()
        nonlocal cur_loss
        nonlocal cur_grad
        cur_loss = loss
        cur_grad = grad_x.clone()

        np_loss = loss.item()
        if prm.metric_type == "grid_edges_scalar":
            np_grad = npu.remove_zeros_in_weight_vector(n,prm.tensor2npy(grad_x),dim).flatten()
        else:
            np_grad = prm.tensor2npy(grad_x.flatten())

        # print("np_grad: ", np_grad)
        # print("Shape np_grad: ", np_grad.shape)
        print("Stats: np_grad: ", npu.stats_on_array(np_grad))

        return np_loss, np_grad

    def lbfgs_sp_callback(xk):

        print("Hello callback !")
        if prm.metric_type == "grid_vertices_tensor_diag":
            x = prm.from_numpy(xk.reshape([dim, N]))
        elif prm.metric_type == "grid_edges_scalar":
            x = prm.from_numpy(npu.insert_zeros_in_weight_vector(n,dim,xk.reshape([dim, -1])))
        else:
            x = prm.from_numpy(xk)

        # TODO: It would be easier to pass x, grad and loss as numpy values, because it is here that we have the info
        #  on var_change, and the metric parametrization, otherwise, we make the silly change : numpy -> torch -> numpy
        save_current_iter(prm.iter_num,cur_loss,0,x,cur_grad,N,loss_array,[],extra_prm)

        prm.iter_num += 1
        # Normally, return True to stop the optimization,
        # but for L-BFGS-B, the callback return is not checked.
        return False

    if var_change:
        bounds = None
    else:
        bounds = [(0, None)] * len(x0)

    # For advanced help on options, see :
    # https://github.com/scipy/scipy/blob/df8c8c35e8a0976e40221df8891ee0b57e05163d/scipy/optimize/lbfgsb_src/lbfgsb.f
    res_x, res_f, res_d = scipy.optimize.fmin_l_bfgs_b(
                    func=numpy_func, # function to minimize
                    x0=x0, # starting estimate
                    fprime=None,    # func also returns the gradient
                    bounds=bounds,
                    m=extra_prm['lbfgs_hist_size'],       # Number of previous gradients used to approximate the Hessian
                    iprint=1,
                    # disp=0,  # Use this to suppress all output
                    maxiter=max_iter,
                    callback=lbfgs_sp_callback,
                    maxls=extra_prm['lbfgs_max_eval']
                    )

    # res = scipy.optimize.minimize(
    #                 fun=numpy_func, # function to minimize
    #                 x0=x0, # starting estimate
    #                 method='L-BFGS-B',  # an order 2 method
    #                 jac=True,           # matching_problems also returns the gradient
    #                 callback=lbfgs_sp_callback,
    #                 bounds=bounds,
    #                 options=dict(
    #                     disp=True,
    #                     maxiter=max_iter,
    #                     maxls=extra_prm['lbfgs_max_eval'],
    #                     maxcor=extra_prm['lbfgs_hist_size']       # Number of previous gradients used to approximate the Hessian
    #                 ))

