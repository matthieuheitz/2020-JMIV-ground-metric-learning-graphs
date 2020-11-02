#!/usr/bin/env python
"""
ml_color_transfer.py

Functions for transferring colors of one image onto another.

"""

import os
import sys

if '--omp_num_threads' in sys.argv:
    os.environ["OMP_NUM_THREADS"] = sys.argv[sys.argv.index('--omp_num_threads')+1]


import numpy as np
import matplotlib
import numpy.linalg

import matplotlib.pyplot as plt
from PIL import Image
import glob
import imageio
import cv2
import json
import re
import time
import argparse

from ml_parameters import *
import np_utils as npu
import ml_core as mlc
import ml_color_timelapse as mlct
import OT_Sinkhorn

# For MacOS
matplotlib.use("agg")   # Non-interactive backend
# matplotlib.use("Qt4Agg") # Interactive backend
# plt.get_current_fig_manager().window.wm_geometry("+1600+400") # For TkAgg
# plt.get_current_fig_manager().window.setGeometry(1600, 400, 1000, 800) # For QtAgg

__author__ = "Matthieu Heitz"


def save_color_hist(A, filebase, scale, color=True, colorspace="RGB", save_plot_png=True, show_plot=False, save_npy=True):

    scale0 = 5000
    if save_npy:
        np.save(filebase,A)
    if save_plot_png or show_plot:
        from mpl_toolkits.mplot3d import Axes3D
        n0, n1, n2 = A.shape
        n = np.cbrt(n0*n1*n2)
        N = n0*n1*n2
        t0 = np.linspace(0,1,n0)
        t1 = np.linspace(0,1,n1)
        t2 = np.linspace(0,1,n2)
        x, y, z, = np.meshgrid(t0,t1,t2,indexing="ij")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Only display values higher than threshold (for speed)
        threshold = 1/N    # mean for a uniform distribution
        T = A >= threshold

        if colorspace == "RGB":
            colors = np.transpose(np.vstack((x.flatten(), y.flatten(), z.flatten())))[T.flatten()] if color else None
        elif colorspace == "LAB":
            C = (np.vstack((x.flatten(), y.flatten(), z.flatten())).T).astype(np.float32)[T.flatten()]
            C[:, 0] *= 100; C[:, 1:] = C[:, 1:] * 255 - 128  # Put in the good range
            colors = cv2.cvtColor(np.expand_dims(C, 0), cv2.COLOR_LAB2RGB)[0]

        ax.scatter(x[T], y[T], z[T], s=(scale*scale0/n)*A[T]/np.max(A), c=colors)
        minmax = (0 - 1/20, 1 + 1/20)   # Add extra border
        ax.set_xlim(minmax); ax.set_ylim(minmax); ax.set_zlim(minmax)
        if colorspace == "RGB":  ax.set_xlabel('R (slow index)'); ax.set_ylabel('G (medium index)'); ax.set_zlabel('B (fast index)');
        elif colorspace == "LAB":  ax.set_xlabel('L (slow index)'); ax.set_ylabel('A (medium index)'); ax.set_zlabel('B (fast index)');

        plt.title("scale=%g, threshold=%.2g, max=%e"%(scale,threshold,np.max(A)))
        if save_plot_png:
            fig.savefig(filebase+".png", bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def save_3d_metric(A, filebase, scale=1, colorspace="RGB", save_plot_png=True, show_plot=False, save_npy=True):

    scale0 = 5000
    if save_npy:
        np.save(filebase,A)
    if save_plot_png or show_plot:
        from mpl_toolkits.mplot3d import Axes3D
        n0, n1, n2 = A.shape
        n = np.cbrt(n0*n1*n2)
        N = n0*n1*n2
        t0 = np.linspace(0,1,n0)
        t1 = np.linspace(0,1,n1)
        t2 = np.linspace(0,1,n2)
        x, y, z, = np.meshgrid(t0,t1,t2,indexing="ij")
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(x, y, z, s=(scale*scale0/n), c=A.flatten(), cmap='coolwarm')
        plt.colorbar(p)
        minmax = (0 - 1/20, 1 + 1/20)   # Add extra border
        ax.set_xlim(minmax); ax.set_ylim(minmax); ax.set_zlim(minmax)
        if colorspace == "RGB":  ax.set_xlabel('R (slow index)'); ax.set_ylabel('G (medium index)'); ax.set_zlabel('B (fast index)');
        elif colorspace == "LAB":  ax.set_xlabel('L (slow index)'); ax.set_ylabel('A (medium index)'); ax.set_zlabel('B (fast index)');

        plt.title("scale=%g"%(scale))
        if save_plot_png:
            fig.savefig(filebase+".png", bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def color_transfer_ot_barycentric_proj(im,im_hist,target_hist,xi,L,color_interp="bin_value",sink_conv_filebase="",
                                       apply_bilateral_filter=False, sigmaRange=0,sigmaSpatial=0, colorspace="RGB"):
    """

    :param im:
    :param im_hist:
    :param target_hist:
    :param xi:
    :param L:
    :param color_interp:
    :param sink_conv_filebase: Save a plot of the Sinkhorn error, unless it's ""
    :param apply_bilateral_filter:
    :param sigmaRange:
    :param sigmaSpatial:
    :return:
    """
    p = im_hist
    q = target_hist
    N = im_hist.size
    n = int(round(np.cbrt(N)))
    if n**3 != N : print("Error: The histogram should be of size n^3")

    # Sinkhorn algorithm
    a = np.zeros(N)
    b = np.ones(N)
    err_p = np.zeros(L)
    err_q = np.zeros(L)
    for i in range(0,L):
        if i%(L/5) == 0: print(i)
        xib = xi(b)
        err_p[i] = np.linalg.norm(a*xib - p)/np.linalg.norm(p)
        if not np.isfinite(err_p[i]):
            print("WARNING: Wild NaN appeared at iter %d. Stopping Sinkhorn here."%i)
            break
        a = p / xib
        xia = xi(a)
        err_q[i] = np.linalg.norm(b*xia - q)/np.linalg.norm(q)
        if not np.isfinite(err_q[i]):
            print("WARNING: Wild NaN appeared at iter %d. Stopping Sinkhorn here."%i)
            break
        b = q / xia
        # print(xia)
        # print(xib)

    # plt.figure(); plt.imshow(np.log(xi(np.eye(N))), cmap='gray') # Metric
    # Pi = np.matmul(np.diag(a), np.matmul(xi(np.eye(N)), np.diag(b)))
    # plt.figure(); plt.imshow(Pi, cmap='gray')
    # plt.figure(); plt.imshow(np.log(Pi), cmap='gray')

    # Plot Sinkhorn error
    if sink_conv_filebase:
        fig = plt.figure()
        plt.semilogy(err_p)
        plt.semilogy(err_q)
        plt.legend(["err_p","err_q"])
        plt.title("Sinkhorn marginal constraint")
        fig.savefig(os.path.join(prm.outdir, sink_conv_filebase+".png"), bbox_inches="tight")
        plt.close(fig)

    # Get the vector of all colors (bin centers)
    t = (np.arange(0, n) + 1/2)/n # Get the bin centers
    cr,cg,cb = np.meshgrid(t,t,t, indexing="ij")
    bin_colors = np.vstack((cr.flatten(),cg.flatten(),cb.flatten())).T
    # new_colors = (np.expand_dims(a,1)*xi(np.expand_dims(b,1)*bin_colors))/np.expand_dims(im_hist,1)
    # new_colors = (np.expand_dims(a,1)*xi(np.expand_dims(b,1)*bin_colors))/np.tile((a*xi(b)),(3,1)).T
    new_colors = (xi(np.expand_dims(b,1)*bin_colors))/np.tile((xi(b)),(3,1)).T
    # print("max new_colors = %f" % np.max(new_colors))
    # print("min new_colors = %f" % np.min(new_colors))
    if np.max(new_colors) > 1:
        print("WARNING: The color values in the new colors exceed 1 : max = %f"%np.max(new_colors))
        new_colors = np.clip(new_colors,0,1)
    h, w = im.shape[0:2]
    imr = im.reshape(-1,3)

    im_ct = np.zeros([h*w,3])

    if color_interp == 'bin_value' :
        # Don't interpolate, just stairs (nearest neighbor)
        ind = (imr*(n-1e-4)).astype('int')  # Remove an epsilon to avoid having the value n
        rav_ind = np.ravel_multi_index(ind.T, (n, n, n))
        im_ct = new_colors[rav_ind]
    elif color_interp == 'linear':
        print("WARNING: This doesn't make sense, I should interpolate with values that are close spatially, in all dimensions,"
              "not just the index before and after (which amounts to interpolating only in the fast speed dimension.")
        exit(-1)
        # Get upper and lower sample index to interpolate between.
        i_low = np.floor(imr*n-1/2).astype('int')
        i_high = np.ceil(imr*n-1/2).astype('int')
        i_low[i_low < 0] = 0 # Border interpolation is flat
        i_high[i_high > n-1] = n-1 # Border interpolation is flat
        i_rav_low = np.ravel_multi_index(i_low.T, (n, n, n))
        i_rav_high = np.ravel_multi_index(i_high.T, (n, n, n))
        # Interpolation weights
        alpha = np.abs(imr - bin_colors[i_rav_low]) / (1 / n)
        im_ct = (1 - alpha) * new_colors[i_rav_low] + alpha * new_colors[i_rav_high]
    else:
        print("ERROR: Unrecognized value for parameter 'color_interp'")
        exit(-1)

    im_out = im_ct.reshape([h,w,3])

    if apply_bilateral_filter:

        # Get file index to save each file
        f_index = sink_conv_filebase.split("-",maxsplit=1)[1]

        im_rgb = im_out
        # Save image before bilteral filtering
        if colorspace == "LAB":     # Convert to RGB before saving image
            # C = (np.vstack((x.flatten(), y.flatten(), z.flatten())).T).astype(np.float32)[T.flatten()]
            im_lab = im_out.copy().astype(np.float32)
            im_lab[:,:,0] *= 100; im_lab[:,:,1:] = im_lab[:,:,1:] * 255 - 128  # Put in the good range
            im_rgb = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)

        # Save the image before the bilateral filter (bbf)
        # imageio.imsave(os.path.join(prm.outdir,"interp-%s-bf.png"%f_index),im_out)
        Image.fromarray((im_rgb * 255).astype('uint8'), 'RGB').save(os.path.join(prm.outdir, "interp-bbf-%s.png"%f_index))

        # Apply the bilateral filter technique
        print("Applying bilateral filter")
        in_filter = im_out - im
        import bilateral_approximation as bf
        edge_min = 0.0
        edge_max = 1.0
        if not sigmaSpatial:
            sigmaSpatial = np.min(in_filter[:,:,0].shape)/16.0
            print("Setting sigmaSpatial = %f"%sigmaSpatial)
        if not sigmaRange:
            sigmaRange = (edge_max - edge_min)/10.0
            print("Setting sigmaRange = %f"%sigmaRange)
        # Cross Bilateral filtering:
        # We smooth the values of the difference btw color-transferred and input image,
        # but respecting the edges of the input image, not of the difference one.
        im_bf = bf.bilateral_approximation_color(in_filter, im, sigmaSpatial, sigmaRange, edgeMin=edge_min, edgeMax=edge_max)
        # Clamp values because negative or > 1 values don't make sense for color.
        im_ct_bf = np.clip(im + im_bf, 0, 1)

        # # For Debug
        # fig = plt.figure(figsize=(16,9))
        # plt.subplot(231)
        # plt.imshow(np.mean(np.abs(in_filter),axis=2)); plt.title('ct - in')
        # plt.colorbar()
        # plt.subplot(232)
        # plt.imshow(im); plt.title('in')
        # plt.subplot(233)
        # plt.imshow(im_out); plt.title('ct')
        # plt.subplot(234)
        # plt.imshow(np.mean(np.abs(im_bf),axis=2)); plt.title('cross_bf(ct - in, in)')
        # plt.colorbar()
        # plt.subplot(236)
        # plt.imshow(im_ct_bf); plt.title('in + cross_bf(ct - in, in)')
        # # plt.show()
        # fig.savefig(os.path.join(prm.outdir,"pipeline-%s.png"%f_index), bbox_inches="tight")
        # plt.close(fig)

        im_out = im_ct_bf

    return im_out


def prolong_metric(w, n, num_prolong, metric_type, sp_order=2):
    """
    Prolongs the metric of a square nd array (3d histogram)
    Prolongation of the histogram is co-located (existing points stay where they are)
    Prolongation of the metric is not, so it's a little more complicated
    :param w: Input array of size [dim, n**(dim-1)*(n-1)]
    :param n: Size of the arrays with which the metric is compatible (all dimensions must have same size: [n,...,n])
    :param num_prolong: Number of times the array should be prolonged.
    :param metric_type: Value of prm.metric_type (see ml_parameters.py)
    :param sp_order: Order of the spline for interpolation.
            A pair (x,y) with x the spline order for interpolation along the axis of the edge,
            and y the spline order for interpolation along other axes.
            Use 0 for constant (nearest) interpolation, 1 for linear, 2 for quadratic, etc...
    :return: the prolonged array, and its new size
    """

    import scipy.ndimage

    dim = w.shape[0]
    n_big = 2 ** num_prolong * (n - 1) + 1

    if metric_type.startswith("grid_vertices"):    # Metric is on points

        # For metric on points, we don't need to differentiate between interpolation along axis of the edge or not,
        # so it's a lot simpler than for metric on edges
        w_big = scipy.ndimage.zoom(w.reshape([dim] + [n, ] * dim), zoom=(1,*((n_big/n,)*dim)), order=sp_order, mode='nearest').reshape([dim,n_big**dim])

    else:                   # Metric is on edges

        # For each vector of edge weight, use a 2x zoom for the axis that the edge is along,
        # and a (2*n-1)/n zoom for the two other axes.
        # This way, we get the correct number of interpolated points in each dimension
        # We interpolate with the same spline order for along the edge axis, and other axes, because
        # our tests showed that it provides better results.
        zero_padded = False
        if w.shape[1] == n**dim:
            w = npu.remove_zeros_in_weight_vector(n,w,dim)
            zero_padded = True
        elif w.shape[1] != n**(dim-1)*(n-1):
            print("ERROR: Metric should be of shape either [dim, n**(dim-1) * (n-1)] = [%d,%d], or [dim, n**dim] = [%d,%d]"%(dim, n ** (dim - 1) * (n - 1),dim, n**dim))
            exit(-1)
        if dim == 3:
            w_b = w[0].reshape([n, n, n - 1])    # Blue (fast axis)
            w_g = w[1].reshape([n, n - 1, n])  # Green (medium axis)
            w_r = w[2].reshape([n - 1, n, n])    # Red (slow axis)
            w_b_big = scipy.ndimage.zoom(scipy.ndimage.zoom(w_b, zoom=(n_big/n, n_big/n, 1), order=sp_order, mode='nearest'), zoom=(1, 1, 2**num_prolong), order=sp_order, mode='nearest')
            w_g_big = scipy.ndimage.zoom(scipy.ndimage.zoom(w_g, zoom=(n_big/n, 1, n_big/n), order=sp_order, mode='nearest'), zoom=(1, 2**num_prolong, 1), order=sp_order, mode='nearest')
            w_r_big = scipy.ndimage.zoom(scipy.ndimage.zoom(w_r, zoom=(1, n_big/n, n_big/n), order=sp_order, mode='nearest'), zoom=(2**num_prolong, 1, 1), order=sp_order, mode='nearest')
            w_big = np.stack((w_b_big.flatten(), w_g_big.flatten(), w_r_big.flatten()))
        elif dim == 2:
            w_h = w[0].reshape([n, n - 1])    # Horizontal (fast axis)
            w_v = w[1].reshape([n - 1, n])    # Vertical (slow axis)
            w_h_big = scipy.ndimage.zoom(scipy.ndimage.zoom(w_h, zoom=(n_big/n, 1), order=sp_order, mode='nearest'), zoom=(1, 2**num_prolong), order=sp_order, mode='nearest')
            w_v_big = scipy.ndimage.zoom(scipy.ndimage.zoom(w_v, zoom=(1, n_big/n), order=sp_order, mode='nearest'), zoom=(2**num_prolong, 1), order=sp_order, mode='nearest')
            w_big = np.stack((w_h_big.flatten(), w_v_big.flatten()))
        else:
            print("Dimension %d not supported"%dim)
            exit(-1)
        # Reinsert zeros if they were there at the beginning
        if zero_padded:
            w_big = npu.insert_zeros_in_weight_vector(n_big, dim, w_big)

    return w_big, n_big


def compute_kernel_from_weight_files(weight_glob, in_param, dim, num_prolong=0, plot_new_metric=False, numpy_io=True):

    # Read the metric
    n_in = in_param['n']
    metric_files = np.sort(glob.glob(weight_glob))
    for f in metric_files:
        if not (os.path.isfile(f) and f.endswith('.npy')):
            print("ERROR: The glob pattern for input metric should only match .npy files. It currently matches '%s'."%f)
            exit(-1)
    num_metric_files = len(metric_files)
    metric = None
    # Load metric
    if num_metric_files == dim :
        metric = np.zeros([dim, n_in**(dim - 1)*(n_in - 1)])
        for i in np.arange(0, num_metric_files):
            metric[i, :] = np.load(metric_files[i]).flatten()
    elif num_metric_files == 1:
        metric = np.load(metric_files[0])
    else:
        print("Warning: The glob pattern for input metrics matches %d files. It should match 1 or dim=%d." % (num_metric_files, dim))
        exit(-1)

    if num_prolong != 0:
        # Prolongation of the metric
        # TODO: Find a way to have sp_order at 2 and no negative values (clamp or interp in log domain)
        metric_big, n_out = prolong_metric(metric, n_in, num_prolong, in_param['metric_type'], 1)
        if not np.all(metric_big > 0):
            print("Error: The metric has negative values after prolongation")
            exit()
        print("Prolongation of the metric from n=%d to n=%d"%(n_in,n_out))
        # Update variables depending on n in the param file
        in_param['n'] = n_out
        in_param['N'] = n_out**dim
        in_param['alpha'] = (n_out-1)**2    # alpha = 1/h**2 and h=1/(n-1)
    else:
        n_out = n_in
        metric_big = metric

    if plot_new_metric:
        # Replot the metric, and its prolongation if it is the case
        if dim == 2: import ml_kinterp2 as ml_save
        elif dim == 3: import ml_color_timelapse as ml_save
        else: print("Dimension %d unsupported"%dim); exit(-1)
        prm.save_frequency = 1
        prm.iter_num = 0
        prm.iter_save = True
        prm.save_graph_plot = False
        prm.plot_loss = False
        ml_save.save_current_iter(0, prm.tensor(0), 0, prm.from_numpy(metric), prm.from_numpy(np.zeros_like(metric)), n_in**dim,[], [], in_param, False, False)
        if num_prolong != 0:
            ml_save.save_current_iter(1, prm.tensor(0), 0, prm.from_numpy(metric_big), prm.from_numpy(np.zeros_like(metric_big)), n_out**dim,[], [], in_param, False, False)

    Ap = prm.tensor(metric_big)

    xi_torch, exp_C, extra_prm = mlc.compute_kernel_torch(Ap, n_out, dim, in_param['apsp_algo'], in_param)

    # Wrap the kernel to get Numpy arrays in and out.
    if numpy_io:    xi = lambda x: prm.tensor2npy(xi_torch(prm.from_numpy(x)))
    else:           xi = xi_torch

    return xi, exp_C.data.numpy(), metric_big


def test_interpolate_with_new_metric():

    dim = 3

    def_prm = {}
    def_prm['only_interp'] = False
    def_prm['use_existing_interp'] = ""
    def_prm['top_outdir'] = "."
    def_prm['num_interp'] = 0
    def_prm['solver_type'] = ""
    def_prm['metric_type'] = "grid_vertices_tensor_diag"
    def_prm['numerical_scheme'] = "backward_euler"
    def_prm['kernel_version'] = "kernel3"
    def_prm['final_outdir'] = ""
    def_prm['disable_tee_logger'] = False
    def_prm['omp_num_threads'] = ""
    def_prm['kernel_check'] = False
    # Parameters for histogram interpolation
    def_prm['hist_interp'] = "input"
    def_prm['num_prolong_hi'] = 0
    def_prm['L_hi'] = 0
    def_prm['t_heat_hi'] = 0.0
    def_prm['k_heat_hi'] = 0
    def_prm['sig_gamma_hi'] = 0.0
    def_prm['gamma_hi'] = 0.0
    def_prm['sink_check_hi'] = True
    # Parameters for color transfer
    def_prm['color_transfer'] = "euclid"
    def_prm['L_ct'] = 0
    def_prm['t_heat_ct'] = 0.0
    def_prm['k_heat_ct'] = 0
    def_prm['sig_gamma_ct'] = 0.0
    def_prm['gamma_ct'] = 0.0
    def_prm['sink_check_ct'] = True
    def_prm['apply_bilateral_filter_ct'] = True
    def_prm['bf_sigmaSpatial_ct'] = 0.0
    def_prm['bf_sigmaRange_ct'] = 0.0

    help_dict = {}
    help_dict['only_interp'] = "Only compute the interpolation, not the color transfer"
    help_dict['use_existing_interp'] = "Folder containing already computed interpolations: 'interp-hist-array-*.npy'"
    help_dict['num_interp'] = "Number of histogram interpolations. If 0, do as many interpolations as there are " \
                              "image inputs (unless there are only 2 inputs, in which case it will default to 10 interpolations)"
    help_dict['solver_type'] = "If specified, overrides the solver with which the input metric was computed"
    help_dict['metric_type'] = "See ml_parameters.py"
    help_dict['numerical_scheme'] = "Numerical scheme to solve the diffusion equation in time: backward_euler or crank_nicolson"
    help_dict['kernel_version'] = "Which kernel to use: kernel3"
    help_dict['top_outdir'] = "Top directory for output, the result will go in a directory named with parameters, in that top directory."
    help_dict['final_outdir'] = "Override 'top_outdir' and writes directly in final_directory"
    help_dict['disable_tee_logger'] = "Set to True to disable duplicating log to file (used for launch scripts that already redirect with '>'"
    help_dict['omp_num_threads'] = "Set the environment variable OMP_NUM_THREADS"
    help_dict['kernel_check'] = "Check the kernel by diffusing a central Dirac"
    # Parameters for histogram interpolation
    help_dict['hist_interp'] = "Method for histogram interpolation: linear, euclid, input"
    help_dict['num_prolong_hi'] = "Number of prolongations before hist interpolation"
    help_dict['L_hi'] = "Number of Sinkhorn iterations for histogram interpolation. When 0, use the value in the metric params"
    help_dict['t_heat_hi'] = "Diffusion time parameter in diffusion equation (for hist_interp='input') for histogram interpolation. " \
                             "When 0, use the value in the metric params"
    help_dict['k_heat_hi'] = "Number of substeps in diffusion solving (for hist_interp='input') for histogram interpolation. " \
                             "When 0, use the value in the metric params"
    help_dict['sig_gamma_hi'] = "Sigma value for the gaussian convolution (for hist_interp='euclid') for histogram interpolation. gamma=2*sigma**2"
    help_dict['gamma_hi'] = "Gamma value for the gaussian convolution (for hist_interp='euclid') for histogram interpolation"
    help_dict['sink_check_hi'] = "Whether to save the plot of Sinkhorn errors for the histogram interpolation"
    # Parameters for color transfer
    help_dict['color_transfer'] = "Method for color transfer: euclid, input"
    help_dict['L_ct'] = "Number of Sinkhorn iterations for color transfer. When 0, use the value in the metric params"
    help_dict['t_heat_ct'] = "Diffusion time parameter in diffusion equation (for hist_interp='input') for color transfer. " \
                             "When 0, use the value in the metric params"
    help_dict['k_heat_ct'] = "Number of substeps in diffusion solving (for hist_interp='input') for color transfer. " \
                             "When 0, use the value in the metric params"
    help_dict['sig_gamma_ct'] = "Sigma value for the gaussian convolution (for hist_interp='euclid') for color transfer"
    help_dict['gamma_ct'] = "Gamma value for the gaussian convolution (for hist_interp='euclid') for color transfer. gamma=2*sigma**2"
    help_dict['sink_check_ct'] = "Whether to save the plot of Sinkhorn errors for the color transfer"
    help_dict['apply_bilateral_filter_ct'] = "Whether to apply a cross bilateral filter after color transfer"
    help_dict['bf_sigmaSpatial_ct'] = "Sigma for the spatial gaussian in the cross bilateral filter. Set to 0 for default"
    help_dict['bf_sigmaRange_ct'] = "Sigma for the image gaussian in the cross bilateral filter. Set to 0 for default"

    # Add some short arguments
    short_args = {}
    short_args['top_outdir'] = 'o'

    # Helper function to add 2 mutualy exclusive boolean optional arguments : --flag and --no-flag
    def add_boolean_opt_arg(parser, def_dict, name, help_str=""):
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

    parser = argparse.ArgumentParser(description="GMLG - Interpolate and transfer colors with an input metric", epilog="")
    parser.add_argument("in_images", type=str, help="Glob pattern matching input images (e.g. \"/path/to/image*.png\")")
    parser.add_argument("in_metric", type=str, help="Input metric (e.g. /path/to/a-metric.npy). There needs to be a '0-parameters.json' file in the same directory.")

    for name in def_prm:
        if type(def_prm[name]) != bool:
            add_general_opt_arg(parser, def_prm, name, type=type(def_prm[name]), help_str=help_dict[name])
        else:
            add_boolean_opt_arg(parser, def_prm, name, help_str=help_dict[name])

    # Post-processing
    args = parser.parse_args()
    args_dict = args.__dict__

    # Create a dict of only parameters that were passed
    passed_prm = {}
    for i in range(1, len(sys.argv)):  # Skip first argument which is this script name
        # Clean argument
        arg = sys.argv[i].lstrip("-")
        if arg.startswith("no-"):   arg = arg[3:]
        # Check if it is a known parameter (if not, it's an argument value, and we ignore it)
        if arg in args.__dict__:
            # Replace it in prm_dict, using value already processed by the parser
            passed_prm[arg] = args.__dict__[arg]

    # Check clashing arguments
    if 'use_existing_interp' in passed_prm:
        for opt in list(passed_prm.keys()):
            if opt.endswith("_hi") or opt == 'hist_interp' or opt == 'num_interp':
                print("ERROR: Can't pass an argument regarding hist_interp ('%s') when use_existing_interp=True"%opt)
                exit()
    if 'only_interp' in passed_prm:
        for opt in list(passed_prm.keys()):
            if opt.endswith("_ct") or opt == "color_transfer":
                print("ERROR: Can't pass an argument regarding color_transfer ('%s') when only_interp=True"%opt)
                exit()

    if args_dict['hist_interp'] == "euclid":
        if 'sig_gamma_hi' in passed_prm and 'gamma_hi' in passed_prm:   print("Can't pass both 'sig_gamma_hi' and 'gamma_hi'."); exit()
        elif 'sig_gamma_hi' in passed_prm:    args_dict['gamma_hi'] = 2*args_dict['sig_gamma_hi']**2
        elif 'gamma_hi' in passed_prm:        args_dict['sig_gamma_hi'] = np.sqrt(args_dict['gamma_hi']/2)
        else: print("Must specify either 'sig_gamma_hi' or 'gamma_hi'"); exit()
    if args_dict['color_transfer'] == "euclid" and not ('only_interp' in passed_prm):
        if 'sig_gamma_ct' in passed_prm and 'gamma_ct' in passed_prm:   print("Can't pass both 'sig_gamma_ct' and 'gamma_ct'."); exit()
        elif 'sig_gamma_ct' in passed_prm:    args_dict['gamma_ct'] = 2*args_dict['sig_gamma_ct']**2
        elif 'gamma_ct' in passed_prm:        args_dict['sig_gamma_ct'] = np.sqrt(args_dict['gamma_ct']/2)
        else: print("Must specify either 'sig_gamma_ct' or 'gamma_ct'"); exit()

    # Positional arguments
    in_images = args_dict['in_images']
    in_metric = args_dict['in_metric']
    # HI arguments
    hist_interp = args_dict['hist_interp']
    num_prolong_hi = args_dict['num_prolong_hi']
    L_hi = args_dict['L_hi']
    t_heat_hi = args_dict['t_heat_hi']
    k_heat_hi = args_dict['k_heat_hi']
    sig_gamma_hi = args_dict['sig_gamma_hi']
    gamma_hi = args_dict['gamma_hi']
    sink_check_hi = args_dict['sink_check_hi']
    # CT arguments
    color_transfer = args_dict['color_transfer']
    L_ct = args_dict['L_ct']
    t_heat_ct = args_dict['t_heat_ct']
    k_heat_ct = args_dict['k_heat_ct']
    sig_gamma_ct = args_dict['sig_gamma_ct']
    gamma_ct = args_dict['gamma_ct']
    sink_check_ct = args_dict['sink_check_ct']
    apply_bilateral_filter_ct = args_dict['apply_bilateral_filter_ct']
    bf_sigmaSpatial_ct = args_dict['bf_sigmaSpatial_ct']
    bf_sigmaRange_ct = args_dict['bf_sigmaRange_ct']
    # Other arguments
    solver_type = args_dict['solver_type']
    use_existing_interp = args_dict['use_existing_interp']
    only_interp = args_dict['only_interp']
    num_interp = args_dict['num_interp']
    top_outdir = args_dict['top_outdir']
    final_outdir = args_dict['final_outdir']
    kernel_check = args_dict['kernel_check']

    # Fun with paths
    if not (os.path.isfile(in_metric) and in_metric.endswith(".npy")):
        print("ERROR: in_metric ('%s') should be an existing metric file with a '.npy' extension."%in_metric)
        exit(-1)
    param_filename = "0-parameters.json"
    in_metric_base = os.path.basename(in_metric)
    in_metric_dir = os.path.dirname(in_metric)
    # Metric ID is composed of the folderpath, then "I" and the iteration number of the metric read as input.
    metric_id = in_metric_dir.replace('/','_') + "I" + re.findall(r'\d+', in_metric_base)[0].lstrip('0')
    in_param = os.path.join(in_metric_dir, param_filename)
    dataset = in_images.rsplit("/", maxsplit=2)[1]

    # Read parameters from param_file
    metric_prm = json.load(open(in_param))
    # Make sure the set of parameters is compatible with current code version
    metric_prm = prm.forward_compatibility_prm_dict(metric_prm)

    # Get local parameters from metric_prm
    colorspace = metric_prm['colorspace']

    # Parameters passed on the command line override those in the metric
    if solver_type:
        metric_prm['solver_type'] = solver_type

    # Histogram interpolation
    if use_existing_interp:
        existing_interp_metric = json.load(open(os.path.join(use_existing_interp, param_filename)))
        # Check if we are using the same metric and images as the ones that were used for those interps
        if existing_interp_metric['in_metric'] != in_metric:
            print("ERROR: Using a metric ('%s') different from the one used to generate the existing interp ('%s')"%(in_metric, existing_interp_metric['in_metric']))
            exit(-1)
        if existing_interp_metric['in_images'] != in_images:
            print("ERROR: Using images different from the ones used to generate the existing interp")
            exit(-1)
        # These parameters won't be used, but the param dict will be consistent
        # with the parameters that generated the interpolation.
        hist_interp = existing_interp_metric['hist_interp']
        n_in = existing_interp_metric['n_in']
        n_hi = existing_interp_metric['n_hi']
        N_hi = existing_interp_metric['N_hi']
        L_hi = existing_interp_metric['L_hi']
        t_heat_hi = existing_interp_metric['t_heat_hi']
        k_heat_hi = existing_interp_metric['k_heat_hi']
        num_prolong_hi = existing_interp_metric['num_prolong_hi']
        sig_gamma_hi = existing_interp_metric['sig_gamma_hi']
        num_interp = existing_interp_metric['num_interp']
        # No need to set metric_prm_hi as it won't be used.
    else:
        metric_prm_hi = metric_prm.copy()
        n_in = metric_prm_hi['n']
        n_hi = 2**num_prolong_hi * (n_in-1) + 1
        N_hi = n_hi**dim
        if hist_interp != "linear":
            # If the parameter is not 0, use that value, else use the value in metric_prm_hi; Harmonize both variables
            L_hi = metric_prm_hi['L'] = L_hi if L_hi else metric_prm_hi['L']
        if hist_interp == "input":  # If 'euclid', kernel is computed through convolutions, else through heat method
            t_heat_hi = metric_prm_hi['t_heat'] = t_heat_hi if t_heat_hi else metric_prm_hi['t_heat']
            k_heat_hi = metric_prm_hi['k_heat'] = k_heat_hi if k_heat_hi else metric_prm_hi['k_heat']

    # Color transfer
    metric_prm_ct = metric_prm.copy()
    n_ct = n_hi
    N_ct = n_ct**dim
    if color_transfer != "linear":
        # If the parameter is not 0, use that value, else use the value in metric_prm_ct; Harmonize both variables
        L_ct = metric_prm_ct['L'] = L_ct if L_ct else metric_prm_ct['L']
    if color_transfer == "input":  # If 'euclid', kernel is computed through convolutions, else through heat method
        t_heat_ct = metric_prm_ct['t_heat'] = t_heat_ct if t_heat_ct else metric_prm_ct['t_heat']
        k_heat_ct = metric_prm_ct['k_heat'] = k_heat_ct if k_heat_ct else metric_prm_ct['k_heat']


    # Output dir
    # Two modes:
    # 1: specify final_outdir, and data will just go there.
    if final_outdir:
        prm.outdir = final_outdir
    # 2: final_outdir is empty and the program generates a name with parameters.
    else:
        # If we use an existing interp, take the same folder name and append the parameters for color transfer
        prm.outdir = top_outdir
        if use_existing_interp:
            prm.outdir = os.path.join(prm.outdir, os.path.basename(use_existing_interp))
        else:
            prm.outdir = os.path.join(prm.outdir, "%s_%s__hi%s_nhi%d_Lhi%d"%(dataset,metric_id,hist_interp,n_hi,L_hi))
            if hist_interp == "euclid":
                prm.outdir += "_ghi%g"%gamma_hi
            if hist_interp == "input":
                prm.outdir += "_thi%0.2e"%t_heat_hi
                prm.outdir += "_Khi%d"%k_heat_hi

        if not only_interp:
            prm.outdir += "__ct%s_nct%d_Lct%d"%(color_transfer,n_ct,L_ct)
            if color_transfer == "euclid":
                prm.outdir += "_gct%g"%gamma_ct
            if color_transfer == "input":
                prm.outdir += "_tct%0.2e"%t_heat_ct
                prm.outdir += "_Kct%d"%k_heat_ct
            if apply_bilateral_filter_ct:
                prm.outdir += "__bf"
                if bf_sigmaSpatial_ct or bf_sigmaRange_ct:
                    prm.outdir += "_sigS%0.2g_sigR%0.2g"%(bf_sigmaSpatial_ct,bf_sigmaRange_ct)

    os.makedirs(prm.outdir, exist_ok=True)

    # By default enabled, so that it uses the logger when launched like 'python <script>',
    # but can be disabled when using a launch script that already redirects output to file.
    logger = None
    if not args_dict['disable_tee_logger']:
        # Logger that duplicates output to terminal and to file
        # This one doesn't replace sys.stdout but just adds a tee.
        log_file_base = "0-alloutput.txt"
        logger = Logger(os.path.join(prm.outdir, log_file_base))

    # Print information after having potentially added the logger
    # The logger has to be put here because we need to know what the outdir is
    print("Dataset: %s"%dataset)
    print("Metric ID: %s"%metric_id)
    print("Histogram size for interpolation:",n_hi)
    print("Histogram size for color transfer:",n_ct)

    # Build kernels if necessary
    if not use_existing_interp:
        if solver_type: prm.solver_type = metric_prm_hi['solver_type']
        # Compute kernel for hist interp
        if hist_interp == "input":
            xi_hi, _, weights = compute_kernel_from_weight_files(in_metric, metric_prm_hi, dim, num_prolong_hi)
        elif hist_interp == "euclid":
            xi_hi = mlc.compute_3d_euclidean_kernel_native(n_hi, gamma_hi)
        else:           # "linear"
            xi_hi = 0
        # Check kernel
        if xi_hi and kernel_check:
            prm.iter_num = 0    # This is to differentiate files written by check_kernel. 0: hi, 1: ct
            mlct.check_kernel(xi_hi,n_hi,N_hi,metric_prm_hi,kernel_io_torch=False)
    else:
        xi_hi = 0

    if not only_interp:
        if solver_type: prm.solver_type = metric_prm_ct['solver_type']
        # Compute kernel for color transfer
        if color_transfer == "input":
            xi_ct, _, weights = compute_kernel_from_weight_files(in_metric, metric_prm_ct, dim, num_prolong_hi)
        elif color_transfer == "euclid":
            # xi_ct = mlc.compute_3d_euclidean_kernel_numpy(n, "convolution_npy", gamma_ct)
            xi_ct = mlc.compute_3d_euclidean_kernel_native(n_ct, gamma_ct)
        else:
            xi_ct = 0
            print("Unrecognized value '%s' for color_transfer"%color_transfer)
            exit()
        # Check kernel
        if xi_hi and kernel_check:
            prm.iter_num = 1    # This is to differentiate files written by check_kernel. 0: hi, 1: ct
        if xi_ct and kernel_check: mlct.check_kernel(xi_ct,n_ct,N_ct,metric_prm_ct,kernel_io_torch=False)
    else:
        xi_ct = 0

    # Read input images
    image_files = np.sort(glob.glob(in_images))
    for f in image_files:
        if not os.path.isfile(f):
            print("ERROR: The glob pattern for input images should only match files, not directories.")
            exit(-1)
    num_in_hist = len(image_files)
    if num_in_hist < 2:
        print("ERROR: The glob pattern for input images matches %d files. It should match at least 2."%num_in_hist)
        exit()
    if num_in_hist > 2:
        print("The glob pattern for input images matches %d files. I will consider only the first and last file."%num_in_hist)

    # Choose number of interpolations
    # If num_interp is non-zero, that's the number of interpolation we want.
    # If it's zero, set to the number of input images. If num_in_hist==2, set to 10.
    if num_interp == 0:
        num_interp = num_in_hist if num_in_hist != 2 else 10

    vmin = 1e-6  # Add minimal mass to histograms (later divided by N)
    normalize = lambda p: p / np.sum(p)
    # Load images
    images = {}     # Allows input images of different sizes
    images_rgb = {}     # Save RGB versions of input images
    hists = np.zeros([num_in_hist, N_hi])
    scales = np.zeros(num_in_hist)
    for i in np.arange(0, num_in_hist):     # num_in_hist is generally 2: Beginning and enf of the interpolation
        images[i] = np.array(imageio.imread(image_files[i]))
        images_rgb[i] = images[i]
        if colorspace == "LAB":
            If32 = images[i].astype(np.float32)/255
            If32_lab = cv2.cvtColor(If32,cv2.COLOR_RGB2LAB)
            If32_lab[:,:,0] *= 255/100; If32_lab[:,:,1:] = If32_lab[:,:,1:] + 128
            images[i] = If32_lab
        hist_item = cv2.calcHist([images[i]], [0, 1, 2], None, [n_hi, n_hi, n_hi], [0, 255, 0, 255, 0, 255])
        scales[i] = np.sum(hist_item)
        hist_item = normalize(hist_item + vmin*np.max(hist_item)/N_hi)
        # save_color_hist(hist_item, os.path.join(prm.outdir, "input-hist-%02d" % i), 1) # Save input histograms
        hists[i] = hist_item.flatten()

    if not only_interp:
        # Also load the first image as a histogram at the good size for color transfer
        u1_h_ct = cv2.calcHist([images[0]], [0, 1, 2], None, [n_ct, n_ct, n_ct], [0, 255, 0, 255, 0, 255])
        u1_h_ct = normalize(u1_h_ct + vmin*np.max(u1_h_ct)/N_ct).flatten()

    # Could have updated it from args in args_dict but doesn't work for manual start, so I'd have to write it twice.
    # Create parameter dictionary
    param_dict = {
        'in_images': in_images,
        'in_metric': in_metric,
        'n_in': n_in,

        'hist_interp': hist_interp,
        'num_prolong_hi': num_prolong_hi,
        'n_hi': n_hi,
        'L_hi': L_hi,
        'sig_gamma_hi': sig_gamma_hi,
        't_heat_hi': t_heat_hi,
        'k_heat_hi': k_heat_hi,
        'metric_type': args_dict['metric_type'],
        'kernel_version': args_dict['kernel_version'],
        'numerical_scheme': args_dict['numerical_scheme'],

        'color_transfer': color_transfer,
        'n_ct': n_ct,
        'L_ct': L_ct,
        'sig_gamma_ct': sig_gamma_ct,
        't_heat_ct': t_heat_ct,
        'k_heat_ct': k_heat_ct,
        'apply_bilateral_filter_ct': apply_bilateral_filter_ct,
        'bf_sigmaSpatial_ct': bf_sigmaSpatial_ct,
        'bf_sigmaRange_ct': bf_sigmaRange_ct,

        'solver_type': solver_type,
        'use_existing_interp': use_existing_interp,
        'only_interp': only_interp,
        'num_interp': num_interp,

        'outdir': prm.outdir,
        'in_param': in_param,
        'num_in_hist': num_in_hist,
        'vmin': vmin,
        'N_hi': N_hi,
        'gamma_hi': gamma_hi,
        'N_ct': N_ct,
        'gamma_ct': gamma_ct,
        'dataset': dataset,
        'metric_id': metric_id,
        'colorspace': colorspace,

        'final_outdir' : final_outdir,
        'disable_tee_logger' : args_dict['disable_tee_logger'],
        'omp_num_threads' : args_dict['omp_num_threads'],
        'kernel_check': kernel_check,
    }
    # Save parameters
    out_param_file = os.path.join(prm.outdir, '0-parameters.json')
    npu.write_dict_to_json(out_param_file, param_dict)

    # We are going to transfer colors of each interpolated histogram on u1
    u1 = images[0]/255

    # Save histograms if we compute HI + CT
    if not use_existing_interp:
        input_hist_filename = "input-hist"
        if colorspace == "LAB": input_hist_filename += "-lab"
        save_color_hist(hists[0].reshape([n_hi,n_hi,n_hi]), os.path.join(prm.outdir, input_hist_filename+"-00-n%d-p"%n_hi), 1, colorspace=colorspace)
        save_color_hist(hists[num_in_hist-1].reshape([n_hi,n_hi,n_hi]), os.path.join(prm.outdir, input_hist_filename+"-00-n%d-q"%n_hi), 1, colorspace=colorspace)

    fig = plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.imshow(images_rgb[0]/255)
    plt.title("First image")
    plt.subplot(122)
    plt.imshow(images_rgb[num_in_hist-1]/255)
    plt.title("Last image")
    fig.savefig(os.path.join(prm.outdir,"0input_images.png"), bbox_inches="tight")

    P = np.vstack((hists[0], hists[-1]))
    t = np.linspace(0, 1, num_interp)
    for i in np.arange(num_interp):
        print("Interpolation %d" % i)
        w = [1 - t[i], t[i]]
        print("Computing interpolation")
        interp_hist_array_filename = "interp-hist-array"
        interp_hist_image_filename = "interp-hist-image"
        if colorspace == "LAB":
            interp_hist_array_filename += "-lab"
            interp_hist_image_filename += "-lab"
        # Read interp from file to avoid recomputing them multiple times
        if use_existing_interp:
            fpath = os.path.join(use_existing_interp, interp_hist_array_filename+"-%02d.npy" % i)
            print("Reading", fpath)
            hist = np.load(fpath).flatten()
        else:
            if hist_interp == "linear":
                hist = ((1 - t[i]) * hists[0] + t[i] * hists[num_in_hist - 1])
            else:
                hist, err_p, err_q = OT_Sinkhorn.compute_sinkhorn_barycenter(P, w, xi_hi, L_hi)
                if (np.any(np.isnan(hist))):
                    print("NaNs appeared in the Sinkhorn barycenter. Aborting barycenter %d"%i)
                    continue
                    # exit()
                if sink_check_hi:
                    # Plot Sinkhorn error
                    fig = plt.figure()
                    plt.semilogy(err_p)
                    plt.semilogy(err_q)
                    plt.legend(["err_p", "err_q"])
                    plt.title("Sinkhorn marginal constraint")
                    sink_conv_file_hi = "sink_conv_hi-%02d" % i if sink_check_hi else ""
                    fig.savefig(os.path.join(prm.outdir, sink_conv_file_hi + ".png"), bbox_inches="tight")
                    plt.close(fig)

        if not use_existing_interp:
            save_color_hist(hist.reshape([n_hi, n_hi, n_hi]), os.path.join(prm.outdir, interp_hist_image_filename+"-%02d" % i), 1, colorspace=colorspace, save_npy=False)
            np.save(os.path.join(prm.outdir, interp_hist_array_filename+"-%02d" % i), hist.reshape([n_hi, n_hi, n_hi]))

        if not only_interp:
            print("Computing transfer")
            sink_conv_file_ct = "sink_conv_ct-%02d" % i if sink_check_ct else ""
            im = color_transfer_ot_barycentric_proj(u1, u1_h_ct, hist, xi_ct, L_ct, sink_conv_filebase=sink_conv_file_ct,
                                                    apply_bilateral_filter=apply_bilateral_filter_ct, sigmaRange=bf_sigmaRange_ct,
                                                    sigmaSpatial=bf_sigmaSpatial_ct, colorspace=colorspace)

            # Save image before bilteral filtering
            if colorspace == "LAB":  # Convert to RGB before saving image
                # C = (np.vstack((x.flatten(), y.flatten(), z.flatten())).T).astype(np.float32)[T.flatten()]
                im_lab = im.copy().astype(np.float32)
                im_lab[:, :, 0] *= 100; im_lab[:, :, 1:] = im_lab[:, :, 1:] * 255 - 128  # Put in the good range
                im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)

            I = Image.fromarray((im * 255).astype('uint8'), 'RGB')
            I.save(os.path.join(prm.outdir, "interp-%02d.png" % i))

    print("Done")



if __name__ == "__main__":

    test_interpolate_with_new_metric()
