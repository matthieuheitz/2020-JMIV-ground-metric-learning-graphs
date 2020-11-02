#!/usr/bin/env python
"""
ml_color_timelapse.py

OT Metric Learning, with autodiff by Pytorch (autograd):
Learn a metric from K color histograms of a timelapse sequence.
We make the strong assumption that the evolution of the histograms
follows a geodesic in the Wasserstein spacec

"""

import os
import sys

if '--omp_num_threads' in sys.argv:
    os.environ["OMP_NUM_THREADS"] = sys.argv[sys.argv.index('--omp_num_threads')+1]
# https://github.com/numpy/numpy/issues/11826#issuecomment-437291454
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"


import numpy as np
import numpy.random
import matplotlib
# print("Matplotlib backend: ", matplotlib.get_backend())
matplotlib.use('agg')
# print("Matplotlib backend: ", matplotlib.get_backend())
import matplotlib.pyplot as plt
import glob
import imageio
import argparse
import cv2
import time
import sys
import json
import threading

import torch
import torch.sparse

from ml_parameters import *
import np_utils as npu
import ml_core as mlc
import ml_optimizers as mlo
import ml_color_transfer as mlct


__author__ = "Matthieu Heitz"


def gradient_descent(P,w,obs,loss_func,x0,L,gamma,l_rate,max_iter,var_change,extra_prm):
    """

    :param P:
    :param w:
    :param obs:
    :param loss_func:
    :param adj_init:
    :param L:
    :param gamma:
    :param l_rate:
    :param max_iter:
    :return:
    """
    assert (P.shape[0] == w.shape[1])  # Same K
    assert (P.shape[1] == obs.shape[1])  # Same N
    assert (w.shape[0] == obs.shape[0])  # Same S
    N = P.shape[1]

    # Apply change of variable before optimization
    if var_change:
        prm.varch_f = lambda x: torch.log(x) # Variable change: Forward - from normal domain to other domain
        prm.varch_b = lambda x: torch.exp(x) # Variable change: Backward - from other domain to normal domain

    x0 = prm.from_numpy(x0)
    x = prm.tensor(prm.varch_f(x0), requires_grad=True)
    t_P = prm.from_numpy(P)
    t_w = prm.from_numpy(w)
    t_obs = prm.from_numpy(obs)
    print('Initial Point :', prm.varch_b(x.detach()))

    if max_iter < 0: max_iter = np.inf
    prev_x = x0 + 1
    i = 0
    prm.iter_num = i
    epsilon = 1e-9

    # Control variables
    loss_array = []
    crit_array = []

    plt.ioff()
    if prm.plot_loss:
        if prm.show_plot:
            plt.ion()

    func_criterion = lambda x, y: torch.norm(prm.varch_b(x.detach()) - prm.varch_b(y.detach())).item()

    while func_criterion(prev_x,x) > epsilon and i <= max_iter:

        # Compute loss and gradient
        t0 = time.time()
        loss = forward_routine(t_P,t_w,t_obs,loss_func,x,L,gamma,extra_prm)
        t1 = time.time()
        print('F: %f, '%(t1-t0), end='',flush=True)
        loss.backward() # Compute gradient on every element of the cost matrix
        t2 = time.time()
        print('B: %f'%(t2-t1))

        with torch.no_grad():

            # Store previous values
            prev_x = x.clone()
            g = x.grad.clone()

            # Update optimized variable
            x -= l_rate * g
            # Zero the gradient
            x.grad.zero_()
            # Compute stopping vriterion
            criterion = func_criterion(prev_x,x)

            save_current_iter(prm.iter_num, loss, criterion, x, g, N, loss_array, crit_array, extra_prm)

        i = i + 1
        prm.iter_num = i

    return 0


def save_current_iter(i, loss, crit, x, g, N, loss_array, crit_array, extra_prm, print_info=True, save_loss_array=True):

    iter_loss = loss.item()
    np_g = prm.tensor2npy(g)
    metric_on_grid_edges = (extra_prm['metric_type'] == "grid_edges_scalar")    # The only case where the weight vector is zero-padded
    y = prm.tensor2npy(prm.varch_b(x))

    if print_info:
        if metric_on_grid_edges: stats_on_array = npu.stats_on_array_nz
        else:                    stats_on_array = npu.stats_on_array
        print('\ni = %d' % i,
              ', Loss = %g'%iter_loss,
              ', crit = %g'%crit,
              ', abs(g) =', stats_on_array(np.abs(np_g.T)),
              ', varch_b(x) =', stats_on_array(np.abs(y.T))
              )

    # Save data
    loss_array.append(iter_loss)
    crit_array.append(crit)

    # Only deals with Numpy arrays, so no need to be in the torch.no_grad()
    if prm.save_frequency:
        if prm.iter_num % prm.save_frequency == 0 or prm.iter_num == 1:
            adjweights_filename = "a-metric"
            if extra_prm['colorspace'] == "LAB":
                adjweights_filename += "-lab"
            if prm.iter_save:
                adjweights_filename += "-%04d" % i

            np.save(os.path.join(prm.outdir, adjweights_filename), y)

            # Save weights in the 3 directions in three 3D arrays visualized with point size
            # w[0] : Weights on the fast (B, z) axis
            # w[1] : Weights on the medium (G, y) axis
            # w[2] : Weights on the slow (R, x) axis
            # w = npu.get_weights_from_3d_adj_matrix(y)
            n = int(round(np.cbrt(N)))
            if metric_on_grid_edges:
                w = npu.remove_zeros_in_weight_vector(n,y,3)
            else:
                w = y
            # Works if w is [3,n**3] or [3,n**2*(n-1)]
            wr = w[2].reshape([- 1, n, n])
            wg = w[1].reshape([n, - 1, n])
            wb = w[0].reshape([n, n, - 1])
            scale = 0.1
            mlct.save_3d_metric(wr, os.path.join(prm.outdir, adjweights_filename + "_0R"), scale, colorspace=extra_prm['colorspace'], save_plot_png=True, show_plot=False, save_npy=True)
            mlct.save_3d_metric(wg, os.path.join(prm.outdir, adjweights_filename + "_1G"), scale, colorspace=extra_prm['colorspace'], save_plot_png=True, show_plot=False, save_npy=True)
            mlct.save_3d_metric(wb, os.path.join(prm.outdir, adjweights_filename + "_2B"), scale, colorspace=extra_prm['colorspace'], save_plot_png=True, show_plot=False, save_npy=True)

    # Plot the loss and save the figure
    loss_suffix = "" if not extra_prm['restart_same_outdir'] else "2"
    if save_loss_array:
        np.save(os.path.join(prm.outdir, "1-loss%s"%loss_suffix), np.array(loss_array))
    if prm.plot_loss:
        fig1 = plt.figure(figsize=(10, 7))
        plt.clf()
        plt.subplot(211)
        plt.plot(loss_array)
        plt.title("Loss evolution")
        plt.subplot(212)
        plt.semilogy(loss_array)
        plt.title('Loss evolution (log-scale)')
        fig1.savefig(os.path.join(prm.outdir, "1-loss_plot%s.png"%loss_suffix), bbox_inches="tight")
        if prm.show_plot:
            plt.pause(1e-6)
        plt.close(fig1)


def check_kernel(xi,n,N,extra_prm, kernel_io_torch=True):

    # Check kernel quality : diffuse a centered Dirac
    dirac_np = np.zeros(N)
    dirac = prm.from_numpy(dirac_np) if kernel_io_torch else dirac_np
    dirac[(n // 2) * (n**2 + n + 1)] = 1
    if 'dirac_1/h**2' in extra_prm and extra_prm['dirac_1/h**2']:
        dirac/= (extra_prm['h']**2)
    # Take a slice in the xy axis
    xid = xi(dirac).reshape([n, n, n])
    dirac_diffused = prm.tensor2npy(xid) if kernel_io_torch else xid
    dirac_diff = dirac_diffused[:,:,n//2]
    if np.any(dirac_diff < 0):
        print("\nERROR: Negative values in kernel result")
        exit(-1)

    fig_k1 = plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(dirac_np.reshape([n, n, n])[:,:,n//2])
    plt.title("Dirac $u_0$")
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(dirac_diff)
    plt.title("u")
    plt.colorbar()
    axsp3 = plt.subplot(223,aspect='equal')
    # plt.imshow(-np.log(dirac_diff))
    from matplotlib import ticker
    loc = ticker.LogLocator()
    plt.contour(dirac_diff, 10, locator=loc, linewidths=0.5, colors='k')
    plt.contourf(dirac_diff, 10, locator=loc)
    plt.title("u (log contours)")
    plt.colorbar()
    axsp4 = plt.subplot(224,aspect='equal')
    plt.contour(-np.log(dirac_diff), 10, linewidths=0.5, colors='k')
    plt.contourf(-np.log(dirac_diff), 10)
    # plt.imshow(np.sqrt(-np.log(dirac_diff / np.max(dirac_diff))))
    plt.title("-log(u)")
    plt.colorbar()
    plt.suptitle("n=%d, t=%0.2e" % (n, extra_prm['t_heat']))
    fig_k1.savefig(os.path.join(prm.outdir, "d-kernel_a-%04d.png"%prm.iter_num), bbox_inches="tight")
    # TODO: Maybe save u, or -log(u) or sqrt(-log(u/u_max)) as a npy file

    # 3D visualization of the diffusion distance
    from mpl_toolkits.mplot3d import Axes3D
    # fig_k2 = plt.figure(figsize=plt.figaspect(0.5))
    fig_k2 = plt.figure(figsize=(10, 5))
    ax = fig_k2.add_subplot(1, 2, 1, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    t = np.linspace(0, 1, n)
    [x, y] = np.meshgrid(t, t)
    ax.plot_surface(x, y, np.sqrt(-np.log(dirac_diff / np.max(dirac_diff))))
    ax.elev = 0
    ax.azim = 0
    plt.title("sqrt(-log(u/max(u))), max(u) = %g" % np.max(dirac_diff))

    ax = fig_k2.add_subplot(1, 2, 2, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    t = np.linspace(0, 1, n)
    [x, y] = np.meshgrid(t, t)
    ax.plot_surface(x, y, -np.log(dirac_diff))
    ax.elev = 0
    ax.azim = 0
    plt.title("-log(u), max(u) = %g" % np.max(dirac_diff))

    plt.suptitle("n=%d, t=%0.2e"%(n,extra_prm['t_heat']))
    fig_k2.savefig(os.path.join(prm.outdir, "d-kernel_b-%04d.png"%prm.iter_num), bbox_inches="tight")
    # plt.show()
    plt.close(fig_k1)
    plt.close(fig_k2)



def forward_routine(P, w, obs, loss_func, A, L, gamma, extra_prm):
    """
    Computes the Wasserstein barycenter of the S histograms in P with weights w.
    Computes the loss between those barycenters and the observations in obs, and returns
    the total sum of the losses.
    Wasserstein barycenter are computed with the cost matrix C, L Sinkhorn iterations,
    and a entropic regularization of gamma.

    :param P: 'Pole' histograms to compute barycenters from
    :type P: torch.Tensor of shape [N,K]
    :param w: Weights to recover each of the observed histogram in obs.
    :type w: torch.Tensor of shape [S,K]
    :param obs: Observed histograms
    :type obs: torch.Tensor of shape [N,S]
    :param loss_func: loss functor to compare a reconstruction with an observed histogram
    :type loss_func: functor
    :param A: adjacency matrix
    :type A: torch.Tensor of shape [N,N]
    :param L: Number of Sinkhorn iterations
    :type L: int
    :param gamma: Entropic regularization parameter
    :type gamma: double
    :return: Sum of losses between the reconstructed barycenter and the observed histograms.
    :rtype torch.Tensor (scalar)
    """

    # TODO : Replace with 'raise ValueError'
    assert(P.shape[0] == w.shape[1]) # Same K
    assert(P.shape[1] == obs.shape[1]) # Same N
    assert(w.shape[0] == obs.shape[0]) # Same S
    S = w.shape[0]
    N = P.shape[1]
    dim = 3
    n = int(round(np.power(N,1/dim)))

    # Apply the backward variable change
    Ap = prm.varch_b(A)

    # If I want to be able to send to forward_routine a non-zeropadded vector of metric on edges,
    # I can use the function insert_zeros_in_weight_vector_torch here.
    # I'll also have to change some things in the regularization part.
    if Ap.shape != (dim,N):
        print("ERROR: The variable vector must be of shape [dim,N]")
        exit(-1)
    Aps = Ap

    # Add the gamma parameter here, because it is passed as a single parameter in all the previous calls,
    # so adding it to the extra_prm before causes a clash of names
    extra_prm['gamma'] = gamma
    # Create cost matrix and kernel operator
    xi, exp_C, _ = mlc.compute_kernel_torch(Aps, n, 3, prm.apsp_algo, extra_prm)

    if extra_prm['kernel_check'] and prm.save_frequency and (prm.iter_num % prm.save_frequency == 0 or prm.iter_num == 1):
        check_kernel(xi,n,N,extra_prm)

    recs = prm.from_numpy(np.zeros([S,N]))
    err_p = np.zeros([S,L])
    err_q = np.zeros([S,L])
    loss = prm.tensor([0.0])  # Total loss

    sinkhorn_batch = False
    t0 = time.time()

    if extra_prm['rec_method'] == "wass_prop":
        # Displacement interpolation with Wasserstein Propagation
        recs[1:-1] = mlc.sinkhorn_displ_interp_wass_prog(P, S, xi, L)
        recs[0] = P[0]
        recs[-1] = P[-1]

    elif extra_prm['rec_method'] == "sink_bary":

        for j in range(S):
            rec, err_p[j], err_q[j] = mlc.sinkhorn_barycenter_torch(P, w[j], xi, L)
            recs[j] = rec.clone()  # Have to do that, because if I put recs[j] instead of rec above, I get an in-place error.
            print("|", flush=True, end="")

            loss_cur = loss_func(rec, obs[j])
            # print(j, loss_cur)
            loss = loss + loss_cur
            # if extra_prm['save_individual_grads']:

    print("\nReconstruction time = %f"%(time.time()-t0))

    if prm.save_frequency:
        if prm.iter_num % prm.save_frequency == 0 or prm.iter_num == 1:

            # Save reconstructions
            t0 = time.time()
            R = prm.tensor2npy(recs)
            if np.any(np.isnan(R)):
                print("ERROR: A wild NaN appeared in the barycenters")
                exit(-1)

            for j in range(S):
                I = np.reshape(R[j], [n, n, n])
                rec_image_name = "rec-image"
                rec_array_name = "rec-array"
                if extra_prm['colorspace'] == "LAB":
                    rec_image_name += "-lab"
                    rec_array_name += "-lab"
                if prm.iter_save:
                    rec_image_name += "-%04d"%prm.iter_num
                    rec_array_name += "-%04d"%prm.iter_num
                if extra_prm['n_seq'] > 1:
                    rec_image_name += "-%02d"%extra_prm['i_seq']
                    rec_array_name += "-%02d"%extra_prm['i_seq']
                rec_image_name += "-%02d"%j
                rec_array_name += "-%02d"%j
                mlct.save_color_hist(I, os.path.join(prm.outdir, rec_image_name), 1, save_npy=False, colorspace=extra_prm['colorspace'])
                np.save(os.path.join(prm.outdir, rec_array_name), I)

            if extra_prm['sinkhorn_check']:
                # Plot constraint errors to check if number of iterations is sufficient
                # Plot the first one, and the middle one, because they are respectively the most and the least convergent ones
                ind1 = 1
                ind2 = S//2+1
                fig_sc = plt.figure(figsize=(16, 8))
                plt.subplot(221)
                plt.semilogy(err_p[ind1])
                plt.title("Bary %d - constraint err inputs"%ind1)
                plt.subplot(222)
                plt.semilogy(err_q[ind1])
                plt.title("Bary %d - constraint err bary"%ind1)
                plt.subplot(223)
                plt.semilogy(err_p[ind2])
                plt.title("Barycenter %d - constraint err inputs"%(ind2))
                plt.subplot(224)
                plt.semilogy(err_q[ind2])
                plt.title("Barycenter %d - constraint err bary"%(ind2))
                sink_check_name = "d-sinkhorn_convergence-%04d"%prm.iter_num
                if extra_prm['n_seq'] > 1:  sink_check_name += "-%02d"%extra_prm['i_seq']
                fig_sc.savefig(os.path.join(prm.outdir, sink_check_name+".png"), bbox_inches="tight")
                plt.close(fig_sc)

            print("Time data save:",time.time()-t0)

    # Add regularization
    if prm.metric_regul != "metric_noregul":

        ro = extra_prm['metric_regul_ro']
        if extra_prm['metric_type'] == "grid_vertices_tensor_diag":
            N_metric = N
            ones = 1
        elif extra_prm['metric_type'] == "grid_edges_scalar":
            N_metric = dim * n ** (dim - 1) * (n - 1)
            ones = prm.from_numpy(npu.insert_zeros_in_weight_vector(n,dim,np.ones([dim,N_metric//dim])))
        else:
            print("ERROR: Unrecognized value for 'metric_type'")
            exit(-1)

        if prm.metric_regul == "metric_regul5":
            ro_laplacian = extra_prm['metric_regul_ro_lap']   # metric_regul5: minimize the variance and the metric laplacian
            ones_regul = ro * (torch.norm(Ap - ones) ** 2)/N_metric
            laplacian_regul = ro_laplacian * mlc.l2norm_of_laplacian_torch(Ap, n, dim, extra_prm)
            print("ones regul: ",ones_regul.item())
            print("laplace regul: ",laplacian_regul.item())
            loss_regul = ones_regul +  laplacian_regul
        else:
            print("Regularization method '%s' unrecognized"%prm.metric_regul)
            exit(-1)

        print("loss barys = %g, loss regul = %g"%(loss,loss_regul))
        loss = loss + loss_regul


    return loss



def lbfgs(P,w,obs,loss_func,x0,L,gamma,max_iter,var_change,extra_prm):

    assert (P.shape[0] == w.shape[1])  # Same K
    assert (P.shape[1] == obs.shape[1])  # Same N
    assert (w.shape[0] == obs.shape[0])  # Same S
    N = P.shape[1]

    # Apply change of variable before optimization
    if var_change:
        prm.varch_f = lambda x: torch.log(x)  # Variable change: Forward - from normal domain to other domain
        prm.varch_b = lambda x: torch.exp(x)  # Variable change: Backward - from other domain to normal domain

    x0 = prm.from_numpy(x0)
    x = prm.tensor(prm.varch_f(x0), requires_grad=True)
    t_P = prm.from_numpy(P)
    t_w = prm.from_numpy(w)
    t_obs = prm.from_numpy(obs)
    print('Initial Point :', prm.varch_b(x.detach()))

    if max_iter < 0: max_iter = np.inf

    # Control variables
    loss_array = []
    crit_array = []

    plt.ioff()
    if prm.plot_loss:
        if prm.show_plot:
            plt.ion()

    func_criterion = lambda x, y: torch.norm(prm.varch_b(x.detach()) - prm.varch_b(y.detach())).item()

    optimizer = torch.optim.LBFGS([x],
                                  history_size=extra_prm['lbfgs_hist_size'],
                                  # max_iter=5,
                                  max_eval=extra_prm['lbfgs_max_eval'],
                                  lr=extra_prm['lbfgs_lr']
                                  )

    # We need a prev_x different from x to enter the while().
    prev_x = x0 + 1
    epsilon = 1e-9
    i = 0
    prm.iter_num = i
    g = 0
    criterion = 0

    def lbfgs_closure():
        optimizer.zero_grad()

        t0 = time.time()
        loss = forward_routine(t_P, t_w, t_obs, loss_func, x, L, gamma, extra_prm)
        t1 = time.time()
        print('F: %f, ' % (t1 - t0), end='', flush=True)
        # pr.enable()
        loss.backward()
        # pr.disable()
        t2 = time.time()
        print('B: %f' % (t2 - t1))

        nonlocal iter_loss, g, criterion
        g = x.grad.clone()
        iter_loss = loss
        criterion = func_criterion(prev_x, x.detach())

        # Assert no nan in grad
        if np.any(np.isnan(prm.tensor2npy(g))):
            print("ERROR: A wild NaN appeared in the gradient")
            exit(-1)

        # Print here to print every evaluation
        print('\ni = %d' % i,
              ', Loss = %g' % loss.item(),
              ', crit = %g' % criterion,
              ', abs(g) =', npu.stats_on_array_nz(np.abs(prm.tensor2npy(g))),
              ', varch_b(x) =', npu.stats_on_array_nz(np.abs(prm.tensor2npy(prm.varch_b(x))))
              )
        # print("Learning rate:",optimizer.state_dict()['param_groups'][0]['lr'])
        # print("Learning rate:",optimizer.state_dict())

        # print("|", end="", flush=True)

        return loss


    # We want to know the loss and gradient at the intial point, so we have to do
    # one evaluation before doing any steps, but we don't want to rewrite almost
    # identical code before the while loop, so we fit it in the loop using some tricks.

    # This stopping criteria works because when arrived at convergence,
    # LBFGS does not update x, so the criterion = 0
    while func_criterion(prev_x, x) > epsilon and i <= max_iter:

        # Store previous value
        prev_x = x.detach().clone()

        if i == 0:  # Evaluate at starting point
            iter_loss = lbfgs_closure()
            prev_x = x0 + 1 # makes the while continue at iteration 1
        else:       # Update optimized variable
            optimizer.step(lbfgs_closure)

        # Print and save current state
        with torch.no_grad():

            # iter_loss, g and criterion are modified by lbfgs_closure()
            # TODO : Maybe define iter_loss=0 at the very beginning ?
            save_current_iter(prm.iter_num,iter_loss,criterion,x,g,N,loss_array,crit_array,extra_prm)

        i = i + 1
        prm.iter_num = i

    # At the end, in LBFGS optimizer, we have access to :
    # num_func_evals, n_iter, old_dirs, old_stps, ro, al



def arg_start():

    # Set default parameters
    def_prm = {}
    def_prm['pat'] = "*.png"
    def_prm['n'] = 16
    def_prm['L'] = 50  # Sinkhorn iterations
    def_prm['sigamma'] = 0.05
    def_prm['learning_rate'] = 1e3  # Learning rate
    def_prm['max_iter'] = 500  # Set negative to disable condition on maximum of iteration
    def_prm['log_domain'] = True
    def_prm['metric_type'] = "grid_vertices_tensor_diag"
    def_prm['numerical_scheme'] = "backward_euler"
    def_prm['kernel_version'] = "kernel3"
    def_prm['loss_num'] = 2
    def_prm['rand_seed'] = 0
    def_prm['colorspace'] = "RGB"
    def_prm['optimizer'] = "lbfgs_sp"  # gd or lbfgs or or lbfgs_sp
    def_prm['init_metric'] = 'ones'  # 'ones' or *.npy file
    def_prm['init_param_file'] = ""
    def_prm['restart_same_outdir'] = False
    def_prm['omp_num_threads'] = ""

    # Parameters in OTMLParameters (prm.*)
    # Overwrite some of the default ones
    def_prm['outdir'] = 'out'
    def_prm['disable_tee_logger'] = False
    def_prm['metric_regul'] = "metric_regul5"  # Regularization of the metric : 'metric_regul5'
    def_prm['apsp_algo'] = 'Numpy_kernel'  # Type of APSP algorithm: 'Numpy_kernel'
    def_prm['solver_type'] = "SparseDirect"
    def_prm['iter_save'] = True
    # Keep other defaults
    def_prm['plot_loss'] = prm.plot_loss
    def_prm['show_plot'] = prm.show_plot
    def_prm['save_frequency'] = prm.save_frequency
    def_prm['dtype'] = prm.dtype.__str__().split('.')[1]
    def_prm['cuda_enabled'] = prm.cuda_enabled  # Careful, this one has a different name
    def_prm['numba_jit_enabled'] = prm.numba_jit_enabled

    # Parameters in extra_prm
    def_prm['sinkhorn_check'] = True
    def_prm['kernel_check'] = False
    def_prm['m_t_heat'] = 0.0
    def_prm['t_heat'] = 1e-2
    def_prm['k_heat'] = 20
    def_prm['lap_norm'] = False
    def_prm['rec_method'] = "sink_bary"  # wass_prop or sink_bary
    def_prm['save_individual_grads'] = False
    def_prm['lbfgs_hist_size'] = 20
    def_prm['lbfgs_max_eval'] = 20
    def_prm['lbfgs_lr'] = 3e3
    def_prm['SD_algo'] = "Cholesky"  # LU or Cholesky
    def_prm['metric_regul_ro'] = 0    # Parameter for the regularization term
    def_prm['metric_regul_ro_lap'] = 3e0    # Parameter for the laplacian regularization term

    help_dict = {}
    help_dict['pat']                    = "Pattern for input images (i.e. '*.png' or '*.jpg')"
    help_dict['n']                      = "Color histogram size (n*n*n)"
    help_dict['L']                      = "number of iterations for the Sinkhorn algorithm"
    help_dict['sigamma']                = "standard deviation for the entropic regularization parameter: gamma = 2*sigamma**2"
    help_dict['learning_rate']          = "Learning rate for gradient descent"
    help_dict['max_iter']               = "Max number of optimizer iterations. Set negative to disable condition on maximum of iteration"
    help_dict['log_domain']             = "Optimize metric coefficients in the log domain (so that they can't become negative)"
    help_dict['loss_num']               = "Loss function to compare reconstructions and input: 1:TV, 2:Q, 3:W, 4:KL"
    help_dict['rand_seed']              = "Random seed for the random number generator"
    help_dict['colorspace']             = "Colorspace in which to compute histograms: RGB, LAB"
    help_dict['optimizer']              = "Optimizer for the metric: gd or lbfgs or or lbfgs_sp"
    help_dict['init_metric']            = "Initialization of the metric: 'ones', or a .npy file"
    help_dict['init_param_file']        = "Initialize all parameters with values from a json file. In that case an empty string ("") should be passed as indir"
    help_dict['restart_same_outdir']    = "Restart the iteration in the same folder. Only valid when init_param_file != "" and init_metric is a .npy file."
    help_dict['omp_num_threads']         = "Set the environment variable OMP_NUM_THREADS"
    help_dict['outdir']                 = "Output directory where data will be saved"
    help_dict['disable_tee_logger']     = "Set to True to disable duplicating log to file (used for launch scripts that already redirect with '>'"
    help_dict['metric_regul']           = "Regularization of the metric : metric_noregul or metric_regul1 or metric_regul2 or metric_regul3"
    help_dict['apsp_algo']              = "Algorithm to compute All-Pair-Shortest-Path: Numpy_kernel"
    help_dict['solver_type']            = "Solver type for the heat equation: SparseDirect"
    help_dict['metric_type']            = "See ml_parameters.py"
    help_dict['numerical_scheme']       = "Numerical scheme to solve the diffusion equation in time: backward_euler or crank_nicolson"
    help_dict['kernel_version']         = "Which kernel to use: kernel3"
    help_dict['iter_save']              = "Save data for each iteration. When false, data is overwritten at every iteration"
    help_dict['plot_loss']              = "Plot the loss and save the plot as an image"
    help_dict['show_plot']              = "Show the evolution of the loss in a pyplot window"
    help_dict['save_frequency']         = "Data is saved at every 'save_frequency' iteration"
    help_dict['dtype']                  = "Tensor data type: float32 or float64"
    help_dict['cuda_enabled']           = "Enable CUDA to compute on GPU"
    help_dict['numba_jit_enabled']      = "Enable JIT with Numba"
    help_dict['sinkhorn_check']         = "Plot the marginal errors of the Sinkhorn algorithm"
    help_dict['kernel_check']           = "Check the kernel by diffusing a central Dirac"
    help_dict['m_t_heat']               = "When > 0, ignores t_heat value and overwrites it with t_heat=m_t_heat*(h**2)"
    help_dict['t_heat']                 = "Diffusion time parameter in heat equation. If m_t_heat != 0, t_heat is set to m_t_heat*(h**2). Indirectly controls the entropic regularization parameter"
    help_dict['k_heat']                 = "Number of substeps in the resolution of the heat equation"
    help_dict['lap_norm']               = "Normalize the laplacian for the heat equation"
    help_dict['rec_method']             = "Method for reconstructing the histograms: sink_bary or wass_prop"
    help_dict['save_individual_grads']  = "Save gradient as individual files (1 per dimension)"
    help_dict['lbfgs_hist_size']        = "Number of iteration in history for computing the Hessian"
    help_dict['lbfgs_max_eval']         = "Max number of linesearch iterations"
    help_dict['lbfgs_lr']               = "Initial step for optimizer=lbfgs"
    help_dict['SD_algo']                = "Algorithm for SparseDirect: LU or Cholesky"
    help_dict['metric_regul_ro']        = "Parameter for the metric_regul (values should be close to 1)"
    help_dict['metric_regul_ro_lap']    = "Parameter for the laplacian regularizer"

    # Add some short arguments
    short_args = {}
    short_args['n'] = 'n'
    short_args['outdir'] = 'o'
    short_args['loss_num'] = 'l'
    short_args['L'] = 'L'
    short_args['save_frequency'] = 'f'

    # Helper function to add 2 mutualy exclusive boolean optional arguments : --flag and --no-flag
    def add_boolean_opt_arg(parser, def_dict, name, help_str):
        if type(def_dict[name]) != bool:
            print("Error: Default value for '%s' should be a boolean.")
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true', help=help_str + ". Default value is '" + str(def_dict[name]) + "'")
        group.add_argument('--no-' + name, dest=name, action='store_false', help=help_str + ". Default value is '" + str(not def_dict[name]) + "'")
        parser.set_defaults(**{name: def_dict[name]})

    # One-liner that adds an optional argument whose default value is in def_dict
    def add_general_opt_arg(parser, def_dict, name, type, help_str):
        if name in short_args:
            parser.add_argument("-" + short_args[name], "--" + name, default=def_dict[name], type=type,
                                help=help_str + ". Default value is '" + str(def_dict[name]) + "'")
        else:
            parser.add_argument("--" + name, default=def_dict[name], type=type,
                                help=help_str + ". Default value is '" + str(def_dict[name]) + "'")

    parser = argparse.ArgumentParser(description="GMLG - Metric learning for histograms on a 3D grid", epilog="")
    parser.add_argument("indir", type=str, help="Input directory where to find the data (image files)")

    for name in def_prm:
        if type(def_prm[name]) != bool:
            add_general_opt_arg(parser, def_prm, name, type=type(def_prm[name]), help_str=help_dict[name])
        else:
            add_boolean_opt_arg(parser, def_prm, name, help_str=help_dict[name])

    # Get passed arguments
    args = parser.parse_args()

    # If a parameter file is passed, use these parameters
    if args.__dict__['init_param_file'] != "":
        # Read parameters from param_file
        input_prm = json.load(open(args.__dict__['init_param_file']))

        # Check whether the outdir is a new one or the one where the metric was found.
        if args.__dict__['restart_same_outdir']:
            # Simulate that I passed disable_tee_logger, so that its potential value of "True" in input_prm is overwritten.
            sys.argv.append("--no-disable_tee_logger")
            if not args.__dict__['init_metric'].endswith(".npy"):
                print("ERROR: To restart a previous run, provide the 'init_metric' at which you wish to restart.")
                exit()
            if os.path.dirname(args.__dict__['init_param_file']) != os.path.dirname(args.__dict__['init_metric']):
                print("ERROR: 'init_param_file' and 'init_metric' should be in the same directory.")
                exit()
            input_prm['outdir'] = os.path.dirname(args.__dict__['init_param_file'])
            # Set iter number to the last one
            prm.iter_num = int(os.path.splitext(os.path.basename(args.__dict__['init_metric']))[0].split('-')[-1])
        else:
            # Reset parameters that don't make sense to be read from prm file.
            # Replace outdir to avoid writing in the same folder
            input_prm['outdir'] = args.__dict__['outdir']
            input_prm['restart_same_outdir'] = args.__dict__['restart_same_outdir']

        # Copy parameters read from file, to param dict.
        prm_dict = input_prm.copy()

        # Then, the passed arguments overwrite those in the file
        for i in range(1, len(sys.argv)):  # Skip first argument which is this script name
            arg = sys.argv[i]
            if not arg.startswith("-"):
                continue
            # Clean argument
            arg = arg.lstrip("-")
            if arg.startswith("no-"):   arg = arg[3:]
            # Check if it is a known parameter (if not, it's an argument value, and we ignore it)
            if arg in args.__dict__:    # If arg is a long parameter name
                # Replace it in prm_dict, using value already processed by the parser
                prm_dict[arg] = args.__dict__[arg]
            elif arg in short_args.values():    # If arg is a short parameter name
                # Get key from value
                prm_name = list(short_args.keys())[list(short_args.values()).index(arg)]
                # Replace value
                prm_dict[prm_name] = args.__dict__[prm_name]
    else:
        # Update the parameter dict with the passed arguments
        prm_dict = args.__dict__.copy()

    # Get variables needed in this function
    indir = prm_dict['indir']
    pat = prm_dict['pat']
    n = prm_dict['n']
    L = prm_dict['L']
    learning_rate = prm_dict['learning_rate']
    max_iter = prm_dict['max_iter']
    log_domain = prm_dict['log_domain']
    loss_num = prm_dict['loss_num']
    optimizer = prm_dict['optimizer']

    # Overwrite parameters in OTMLParameters
    prm.outdir = prm_dict['outdir']
    prm.metric_type = prm_dict['metric_type']
    prm.metric_regul = prm_dict['metric_regul']
    prm.apsp_algo = prm_dict['apsp_algo']
    prm.solver_type = prm_dict['solver_type']
    prm.iter_save = prm_dict['iter_save']
    prm.plot_loss = prm_dict['plot_loss']
    prm.show_plot = prm_dict['show_plot']
    prm.save_frequency = prm_dict['save_frequency']
    prm.dtype = prm_dict['dtype']
    prm.cuda_enabled = prm_dict['cuda_enabled']
    prm.numba_jit_enabled = prm_dict['numba_jit_enabled']

    # Create output directory
    os.makedirs(prm.outdir, exist_ok=True)

    # By default enabled, so that it uses the logger when launched like 'python <script>',
    # but can be disabled when using a launch script that already redirects output to file.
    logger = None
    if not prm_dict['disable_tee_logger']:
        # Logger that duplicates output to terminal and to file
        # This one doesn't replace sys.stdout but just adds a tee.
        log_file_base = "0-alloutput.txt" if not prm_dict['restart_same_outdir'] else "0-alloutput2.txt"
        logger = Logger(os.path.join(prm.outdir, log_file_base))

    # Warn if reading parameters from file
    if prm_dict['init_param_file'] != "":
        print("IMPORTANT: Reading parameters from:\n%s\n"%prm_dict['init_param_file'])

    # Warn if m_t_heat and t_heat have been passed together
    if "--m_t_heat" in sys.argv and "--t_heat" in sys.argv :
        print("WARNING: m_t_heat and t_heat are both specified. Only m_t_heat will be considered.")

    print("Input dir: %s"%indir)
    print("Output dir: %s"%prm.outdir)

    # Read interpolations
    files = np.sort(glob.glob(os.path.join(indir,pat)))
    n_obs = len(files)
    if n_obs == 0:
        print("ERROR: Found 0 files that match '%s'"%os.path.join(indir,pat))
        exit(-1)

    # Check how many observations there are
    basenames = [os.path.basename(os.path.splitext(f)[0]) for f in files]
    num_num_basenames = np.array([len([int(s) for s in b.split('-') if s.isdigit()]) for b in basenames]) # number of numbers in each basenames
    if not np.all(num_num_basenames == num_num_basenames[0]):
        print("ERROR: Inconsistency in input filenames. Can't determine number of observations.")
        exit(-1)
    indices = np.array([[int(s) for s in b.split('-') if s.isdigit()] for b in basenames])
    if indices.shape[1] == 2:   # If there are two indices in each file, that means we have multiple observations.
        n_seq = len(np.unique(indices[:,0]))
        n_interp = np.count_nonzero(indices[:,0] == indices[0,0])
    else:
        n_seq = 1
        n_interp = indices.shape[0]

    if n_seq*n_interp != len(files):
        print("ERROR: All observations don't have equal numbers of interpolations.")
        exit(-1)

    # Process some arguments and add more in prm_dict
    dim = 3
    N = n**dim
    h = 1/(n-1)
    prm_dict['N'] = N
    prm_dict['alpha'] = 1/h**2
    prm_dict['dim'] = dim
    prm_dict['n_interp'] = n_interp
    prm_dict['n_seq'] = n_seq
    if prm_dict['dtype'] == 'float64': prm.dtype = torch.float64
    if prm_dict['dtype'] == 'float32': prm.dtype = torch.float32
    if prm_dict['cuda_enabled']: prm.enable_cuda()
    prm_dict['device'] = prm.device.__str__()
    gamma = 2*(prm_dict['sigamma'])**2
    prm_dict['gamma'] = gamma
    if prm_dict['m_t_heat'] != 0: prm_dict['t_heat'] = prm_dict['m_t_heat'] * (1 / (n - 1) ** 2)
    loss_options = {1: mlc.loss_TV, 2: mlc.loss_L2, 3: mlc.loss_W, 4: mlc.loss_KL}
    torch.manual_seed(prm_dict['rand_seed'])
    dataset = indir.rsplit("/", maxsplit=2)[-1]
    prm_dict['dataset'] = dataset

    # Save parameters
    param_file_base = "0-parameters.json" if not prm_dict['restart_same_outdir'] else "0-parameters2.json"
    param_file = os.path.join(prm.outdir, param_file_base)
    npu.write_dict_to_json(param_file, prm_dict)

    obs = np.zeros([n_obs,N])
    scales = np.zeros(n_obs)
    vmin = 1e-6
    normalize = lambda p: p / np.sum(p)
    print()
    for i in range(n_obs):
        print("Reading %s"%files[i])
        I = np.array(imageio.imread(files[i]))
        if prm_dict['colorspace'] == "LAB":
            If32 = I.astype(np.float32)/255
            If32_lab = cv2.cvtColor(If32,cv2.COLOR_RGB2LAB)
            If32_lab[:,:,0] *= 255/100; If32_lab[:,:,1:] = If32_lab[:,:,1:] + 128
            I = If32_lab
        hist_item = cv2.calcHist([I], [0,1,2], None, [n,n,n], [0, 255, 0, 255, 0, 255])
        scales[i] = np.sum(hist_item)
        hist_item = normalize(hist_item + vmin*np.max(hist_item)/N)
        obs[i] = hist_item.flatten()
        # mlct.save_color_hist(hist_item,os.path.join(prm.outdir,"hist-input"+"-%03d"%i),1,save_npy=False,colorspace=prm_dict['colorspace'])
    print()

    # Read marginals (same as first and last observations)
    P = obs[np.vstack((np.arange(0, n_obs, n_interp), np.arange(0, n_obs, n_interp) + n_interp - 1)).T.flatten()]

    # Interpolation weights
    interp = np.linspace(0, 1, n_interp)
    # w = np.array([1-interp,interp]).T
    w = np.tile(np.array([1 - interp, interp]).T, (n_seq, 1))

    if prm.apsp_algo != "Numpy_kernel":
        print("The APSP algorithm %s is not recognized."%prm.apsp_algo)
        exit(-1)

    # Initial metric
    x0 = 0
    # Uniform weights
    # w[0] : Weights on the fast (B, z) axis
    # w[1] : Weights on the medium (G, y) axis
    # w[2] : Weights on the slow (R, x) axis
    init_metric = prm_dict['init_metric']
    if init_metric == 'ones':
        if prm_dict['metric_type'] == "grid_edges_scalar":
            w0 = np.ones([3, n**2 * (n - 1)])
            x0 = npu.insert_zeros_in_3d_weight_vector(n, w0)
        elif prm_dict['metric_type'] == "grid_vertices_tensor_diag":
            x0 = np.ones([3, N])
    elif init_metric.endswith(".npy"):
        print("IMPORTANT: Initializing metric with:\n%s\n"%init_metric)
        x0 = np.load(init_metric)
    else:
        print("Initialization of weights '%s' not recognized" % init_metric)
        exit(-1)

    if optimizer == "gd":
        gradient_descent(P, w, obs, loss_options[loss_num], x0, L, gamma, learning_rate, max_iter, log_domain, prm_dict)
    elif optimizer == "lbfgs":
        lbfgs(P, w, obs, loss_options[loss_num], x0, L, gamma, max_iter, log_domain, prm_dict)
    elif optimizer == "lbfgs_sp":
        mlo.lbfgs_scipy(P, w, obs, loss_options[loss_num], x0, L, gamma, max_iter, log_domain, forward_routine, save_current_iter, prm_dict)
    else:
        print("Optimizer '%s' not recognized"%optimizer)
        exit(-1)


    return 0



if __name__ == "__main__":

    arg_start()
    # manual_start()
