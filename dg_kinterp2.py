#!/usr/bin/env python
"""
dg_kinterp2.py

Data Generation for the application ml_kinterp2
This script only does the forward computation of the barycenters, using the metric parameters.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os
from scipy import sparse
import math
import time
import sys

import OT_Sinkhorn

from ml_parameters import *
import np_utils as npu
import ml_core as mlc
import ml_kinterp2 as mlk

# matplotlib.use("Qt5Agg")

__author__ = "Matthieu Heitz"

# Analytic functions to generate a metric

def gaussians3_0(x,y):
    mu = np.array([[1.3, 1], [-1.3, 1], [-0.0, -1.3]])
    sigma = 0.5*np.ones(3)
    g = np.zeros_like(x)
    for i in range(sigma.size):
        g += npu.gaussian2d(x,y,mu[i],sigma[i])
    return g


def gaussian1_0(x,y):
    return npu.gaussian2d(x,y,np.array([0, 0]),np.sqrt(2)/2)


def gaussian1_1(x,y):
    return npu.gaussian2d(x,y,np.array([0, 0]),2)


def gaussian1_2(x,y):
    return npu.gaussian2d(x,y,np.array([0, 0]),1)


def gaussians2_0(x,y):
    return npu.gaussian2d(x,y,np.array([0, -0.7]),1) + npu.gaussian2d(x,y,np.array([0, 0.7]),1)


def gaussians2_1(x,y):
    return npu.gaussian2d(x,y,np.array([0, -0.6]),0.5) + npu.gaussian2d(x,y,np.array([0, 0.6]),0.5)


def gaussians2_2(x,y):
    return npu.gaussian2d(x,y,np.array([0, -3]),1) + npu.gaussian2d(x,y,np.array([0, 3]),1)


def mgaussian1_1(x,y):
    return npu.multidim_gaussian_angle(x,y,[0,0],0.6,6,math.pi/4)


def mgaussian1_2(x,y):
    return npu.multidim_gaussian_angle(x,y,[0,0],0.6,6,-math.pi/4)


def mgaussian2_1(x,y):
    return npu.multidim_gaussian_angle(x,y,[0,0],0.6,6,math.pi/4) - 0.06*npu.gaussian2d(x,y,[0,0],1)


def mgaussian2_2(x,y):
    return npu.multidim_gaussian_angle(x,y,[0,0],0.6,6,-math.pi/4) - 0.06*npu.gaussian2d(x,y,[0,0],1)



# Create output directory
prm.outdir = "test_data"
if not os.path.isdir(prm.outdir):
    os.mkdir(prm.outdir)

# Parameters to use if we want to get metric values from a picture
metric_from_pic = ""
from_pic_im_complement = True
from_pic_interp_order = 0   # Order of interpolation when performing bilinear interpolation of an image to fit dimension to the metric

# Initialize parameters
rand_seed = 4
np.random.seed(rand_seed)
n = 50 # Histogram size
dim = 2
N = n**dim
h = 1/(n-1)
alpha = 1/h**2
m = 5 # Number of points / Gaussians
gen_hist = 'stick_tl_br'     # points, gaussians, 2borders_lr, 4borders, border_tl_br corner_tl_br, corner_tl_br_thick, 1dot_2borders, custom, stick_tl_br
sigpix_gamma = 4        # For euclidean metric
gamma = 2*(sigpix_gamma/n)**2   # For euclidean metric
L = 50 # Sinkhorn iterations
# sigma_pix = 0.5 # Define std dev in pixel scale
# sigma = sigma_pix/n # Rescale sigma for the [0,1] scale (for Gaussian2D)
sigma = 0.05 # Define std dev in image scale, for gaussians in inputs
vmin = 1e-6 # Minimal mass (as a ratio of max value)
k = 11 # Number of interpolations
gen_metric = 'analytic'       # random, old_euclidean, 2param, analytic, from_pic, no_ot_linerp
apsp_algo = 'Numpy_kernel'
solver_type = "SparseDirect"
prm.save_graph_plot = False
sink_check = True
kernel_check = False

# Metric values
metric_ratio_min_max = 1000  # Will be squared if metric is squared
metric_avg = 1  # Average edge length

# APSP-dependant or metric-dependant parameters
# analytic_func = [gaussian1_1] # npu.peaks, gaussian1_0, gaussian1_1, gaussians3_0, gaussians2_0, gaussians2_1
analytic_func = [mgaussian1_1,mgaussian1_2]
lx = 1 # 2param : x-axis cost
ly = 1 # 2param : y-axis cost
theta = 0  # 2param : Angle of the axes

# Define extra parameters for Numpy_kernel
extra_param = {}
extra_param['t_heat'] = 3e-3
extra_param['k_heat'] = 100
extra_param['lap_norm'] = False
extra_param['metric_type'] = "grid_vertices_tensor_diag"
extra_param['kernel_version'] = "kernel3"
extra_param['numerical_scheme'] = "backward_euler"  # backward_euler or crank_nicolson
extra_param['rec_method'] = "sink_bary"  # wass_prop, sink_bary
extra_param['save_individual_grads'] = False
extra_param['SD_algo'] = "Cholesky"  # LU, Cholesky
extra_param['n'] = n
extra_param['dim'] = dim
extra_param['N'] = N
extra_param['alpha'] = alpha

# Whether metric is on points or edges
metric_on_grid_vertices = (extra_param['metric_type'] == "grid_vertices_tensor_diag")
if not metric_on_grid_vertices and extra_param['metric_type'] != "grid_edges_scalar":
    print("ERROR: Metric type '%s' is unsupported.")
    exit(-1)

# Custom points for gen_hist
# Store them in string, otherwise it messes up the json.
# Dataset 2DG-A
# points_p = "[[0.2,0.8]]"
# points_q = "[[0.8,0.2]]"
# Dataset 2DG-B : Choice between vertical or horizontal movement
# points_p = "[[0.2,0.8],[0.8,0.2]]"
# points_q = "[[0.2,0.2],[0.8,0.8]]"
# Dataset 2DG-C : Choice between diagonal ou straight movement
# points_p = "[[0.2,0.5],[0.8,0.5]]"
# points_q = "[[0.5,0.2],[0.5,0.8]]"
# Dataset 2DG-D : Choice between diagonal up or down movement
# points_p = "[[0.2,0.5],[0.5,0.8]]"
# points_q = "[[0.5,0.2],[0.8,0.5]]"
# Dataset
# points_p = "[[0.49,0.15]]"
# points_q = "[[0.49,0.85]]"
# Dataset
points_p = "[[0.2,0.2]]"
points_q = "[[0.8,0.8]]"
# Dataset
# points_p = "[[0.15,0.15]]"
# points_q = "[[0.85,0.85],[0.15,0.85],[0.85,0.15]]"

# Save parameters
param_dict = {}
param_dict['rand_seed'] = rand_seed
param_dict['m'] = m
param_dict['gen_hist'] = gen_hist
param_dict['sigpix_gamma'] = sigpix_gamma
param_dict['gamma'] = gamma
param_dict['L'] = L
param_dict['sigma'] = sigma
param_dict['vmin'] = vmin
param_dict['k'] = k
param_dict['gen_metric'] = gen_metric
param_dict['apsp_algo'] = apsp_algo
param_dict['outdir'] = prm.outdir
param_dict['metric_from_pic'] = metric_from_pic
param_dict['from_pic_im_complement'] = from_pic_im_complement
param_dict['from_pic_interp_order'] = from_pic_interp_order
param_dict['metric_ratio_min_max'] = metric_ratio_min_max
param_dict['metric_avg'] = metric_avg

if gen_hist == "custom":
    param_dict['points_p'] = points_p
    param_dict['points_q'] = points_q

if gen_metric == 'analytic':
    param_dict['analytic_func'] = [f.__name__ for f in analytic_func]
if gen_metric == '2param':
    param_dict['lx'] = lx
    param_dict['ly'] = ly
    param_dict['theta'] = theta

param_dict.update(extra_param)
param_file_fp = os.path.join(prm.outdir, '0-parameters.json')
npu.write_dict_to_json(param_file_fp,param_dict)

t = np.arange(0,n)/(n-1)
# t = np.arange(0,n)/n    # Legacy code to recreate datasets prior to 24/06/19
[x,y] = np.meshgrid(t,t)
Gaussian2D = lambda p0,sigma: npu.gaussian2d(x,y,p0,sigma)
normalize = lambda p: p/np.sum(p)


# Datasets
xp = np.random.rand(m,2)
xq = np.random.rand(m,2)
p = np.zeros([n,n])
q = np.zeros([n,n])
if gen_hist == 'gaussians':
    for i in range(m):
        p += Gaussian2D(xp[i],sigma)
        q += Gaussian2D(xq[i],sigma)
elif gen_hist == 'points':
    for i in range(m):
        dxp = np.round(xp[i]*(n-1))
        dxq = np.round(xq[i]*(n-1))
        p[int(dxp[1]),int(dxp[0])] += 1
        q[int(dxq[1]),int(dxq[0])] += 1
elif gen_hist == '2borders_lr':
    p[:,0] = 1
    q[:,-1] = 1
elif gen_hist == 'corner_tl_br':
    p[0:int(n/2), 0] = 1
    p[0, 0:int(n/2)] = 1
    q[int(n/2):n, -1] = 1
    q[-1, int(n/2):n] = 1
elif gen_hist == 'corner_tl_br_thick':
    thick = int(n/10)
    p[0:int(n/2), 0:thick] = 1
    p[0:thick, 0:int(n/2)] = 1
    q[int(n/2):n, -thick:] = 1
    q[-thick:, int(n/2):n] = 1
elif gen_hist == "stick_tl_br":
    thick = int(n/10)
    off = thick
    p[off:off+int(n/2), off:off+thick] = 1
    q[int(n/2)-off:n-off, -thick-off:-off] = 1
elif gen_hist == '4borders':
    p[:,0] = 1
    p[0,:] = 1
    q[:,-1] = 1
    q[-1,:] = 1
elif gen_hist == '1dot_2borders':
    p[0:4,0:4] = 1 # Can't make it smaller otherwise 100 Sinkhorn iterations make NaNs
    q[:,-1] = 1
    q[-1,:] = 1
elif gen_hist == 'border_tl_br':
    p[:,0] = p[0,:] = 1
    q[:,-1] = q[-1,:] = 1
elif gen_hist == 'custom':
    points_p = eval(points_p)
    points_q = eval(points_q)
    p = np.zeros([n,n])
    q = np.zeros([n,n])
    for i in range(len(points_p)):
        p += Gaussian2D(points_p[i],sigma)
    for i in range(len(points_q)):
        q += Gaussian2D(points_q[i],sigma)
else:
    print("Histogram generation mode not recognized")

# Add minimal mass
p = normalize( p+np.max(p)*vmin/N)
q = normalize( q+np.max(q)*vmin/N)

# print(p)
# print(q)

# pxp = n*xp # Plot where gaussian centers are
# pxq = n*xq # Plot where gaussian centers are
# pxp = (n-1)*xp # Plot where points are
# pxq = (n-1)*xq # Plot where points are
# Show input histograms
# plt.ion()
# plt.figure()
# plt.subplot(121)
# plt.imshow(p,cmap='gray')
# # plt.plot(pxp[:,0],pxp[:,1],'o',mew=sigma*n*100) # Plot where gaussian centers are
# plt.title('Input $p$')
# plt.subplot(122)
# plt.imshow(q,cmap='gray')
# # plt.plot(pxq[:,0],pxq[:,1],'o',mew=sigma*n*100) # Plot where gaussian centers are
# plt.title('Input $q$')

# plt.show()
# Save input histograms
np.save(os.path.join(prm.outdir,'input_p'),p)
np.save(os.path.join(prm.outdir,'input_q'),q)
imageio.imsave(os.path.join(prm.outdir,'input_q.png'),(q/np.max(q)*255).astype('uint8'))
imageio.imsave(os.path.join(prm.outdir,'input_p.png'),(p/np.max(p)*255).astype('uint8'))


if metric_on_grid_vertices:
    n1 = n
    n2 = n
else:
    n1 = n - 1
    n2 = n

# Just generate linear interpolations of the two input histograms to create a fake dataset.
if gen_metric == "no_ot_linerp":

    barys = np.zeros([k, n, n])
    for i in range(k):
        alpha = i / (k - 1)
        interp = (1-alpha)*p + alpha*q
        I = np.reshape(interp, [n, n])
        plt.imshow(I, cmap='gray')
        barys[i] = I

        beta = 1.0 / np.max(I)
        # print(beta)
        imageio.imsave(os.path.join(prm.outdir, "image-%02d.png" % i), (I * beta*255).astype('uint8'))
        np.save(os.path.join(prm.outdir, "array-%02d" % i), I)

    scale = np.max(barys)
    for i in range(k):
        imageio.imsave(os.path.join(prm.outdir, "image-globalscale-%02d.png" % i), (barys[i] / scale*255).astype('uint8'))

    sys.exit(0)



### Build the metric ###
if gen_metric == 'old_euclidean':

    # Euclidian metric
    A = np.array([x.flatten(), y.flatten()])
    # B = np.dot(M,A)
    B = A
    X = np.tile(B[0],(N,1))
    Y = np.tile(B[1],(N,1))
    E = (X-X.T)**2+(Y-Y.T)**2
    C = E
    exp_C = np.exp(-C / gamma)
    # plt.figure()
    # plt.hist(C.flatten(), n ** 2)

    # Create the kernel operator with matrix vector multiplication
    xi = lambda x: np.dot(exp_C, x)
    print("Kernel non-zeros : ", 100 * np.count_nonzero(exp_C) / n ** 4, "%")

    xi_torch = lambda x : prm.from_numpy(xi(prm.tensor2npy(x)))


else:

    horiz_weights = 0
    vert_weights = 0

    if gen_metric == '2param':

        # # Covariance matrix
        # cos2=math.cos(theta)*math.cos(theta)
        # sin2=math.sin(theta)*math.sin(theta)
        # sincos=math.sin(theta)*math.cos(theta)
        # M = np.array([[lx*cos2 + ly*sin2, (lx-ly)*sincos],
        #               [(lx-ly)*sincos, lx*sin2 + ly*cos2]])
        # print(M)
        #
        # # Create the cost matrix for a 2D histogram
        # # X = np.dot(M,(np.ones((2,n1*n2))/n))  # Apply metric transformation to regular values
        # X = np.dot(M,(np.ones((2,n1*n2))/n))  # Apply metric transformation to regular values
        # horiz_weights = X[0]
        # vert_weights = X[1]

        X = np.ones([2,n1*n2])  # Apply metric transformation to regular values
        horiz_weights = X[0]*lx
        vert_weights = X[1]*ly

    if gen_metric == 'random':

        X = np.random.rand(2,n1*n2)
        import scipy.ndimage.filters
        X[0] = scipy.ndimage.filters.gaussian_filter(X[0].reshape(n,n-1),1).reshape(n1*n2)
        X[1] = scipy.ndimage.filters.gaussian_filter(X[1].reshape(n,n-1),1).reshape(n1*n2)
        X = 1 - (X-X.min())/(X.max()-X.min())*(1-1/metric_ratio_min_max)
        X = X/np.mean(X)
        horiz_weights = X[0]
        vert_weights = X[1]

    if gen_metric == 'analytic':

        # Grid sampling
        t1 = np.linspace(-3, 3, n1)
        t2 = np.linspace(-3, 3, n2)
        # Cropped closer to the center
        # t1 = np.linspace(-2, 2, n1)
        # t2 = np.linspace(-2, 2, n2)

        if len(analytic_func) == 1:
            analytic_func.append(analytic_func[0])

        # Sample the edge lengths from the function
        x, y = np.meshgrid(t1, t2)
        dx = analytic_func[0](x,y)
        x, y = np.meshgrid(t2, t1)
        dy = analytic_func[1](x,y)
        # x, y = np.meshgrid(t2, t2)
        # d = analytic_func(x,y)
        # Plot before rescale
        fig = plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.imshow(dx, cmap='coolwarm')
        plt.colorbar()
        plt.title("Horizontal metric")
        plt.subplot(122)
        plt.imshow(dy, cmap='coolwarm')
        plt.colorbar()
        plt.title("Vertical metric")
        fig.savefig(os.path.join(prm.outdir, "a-analytic_func.png"), bbox_inches="tight")
        plt.close(fig)

        # Rescale values
        sdx = 1-(dx-np.min(dx))/np.max(dx-np.min(dx))*(1-1/metric_ratio_min_max)
        sdy = 1-(dy-np.min(dy))/np.max(dy-np.min(dy))*(1-1/metric_ratio_min_max)
        # sd = 1-(d-np.min(d))/np.max(d-np.min(d))*(1-1/metric_ratio_min_max)
        sdx = sdx/np.mean(sdx)*metric_avg
        sdy = sdy/np.mean(sdy)*metric_avg
        # sd = sd/np.mean(sd)*metric_avg

        horiz_weights = sdx.flatten()
        vert_weights = sdy.flatten()


    if gen_metric == 'from_pic':

        Im = imageio.imread(metric_from_pic)
        I = np.array(Im,dtype='float64')/255.0
        I = np.mean(I,axis=2)   # Get gray by averaging R,G and B
        if from_pic_im_complement: I = 1 - I
        if I.shape != (n, n):   # Readjust to the good size
            import scipy.ndimage
            Is = scipy.ndimage.zoom(I,(n/I.shape[0],n/I.shape[1]),order=from_pic_interp_order,mode="nearest")
        else:
            Is = I
        In = 1 - (Is-Is.min())/(Is.max()-Is.min())*(1-1/metric_ratio_min_max)
        X = In/np.mean(In)
        horiz_weights = X if metric_on_grid_vertices else X[:,:-1]
        vert_weights = X if metric_on_grid_vertices else X[:-1,:]


    if metric_on_grid_vertices:
        weights = np.stack((horiz_weights,vert_weights))
    else:
        weights = npu.insert_zeros_in_weight_vector(n,2,np.stack((horiz_weights,vert_weights)))

    adjmat_filename = "a-metric"
    adjgraph_filename = "adj_graph"
    np.save(os.path.join(prm.outdir, adjmat_filename), weights)

    n = int(round(np.sqrt(N)))
    fig2 = plt.figure(figsize=(14, 5))
    fig2.clf()
    if metric_on_grid_vertices:
        z = weights.reshape(-1, n, n)
    else:
        z = npu.remove_zeros_in_weight_vector(n, weights, 2).reshape(-1, n, n - 1)
    plt.subplot(121)
    plt.imshow(z[0], cmap='coolwarm')
    plt.title("Horizontal metric")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(z[1], cmap='coolwarm')
    plt.title("Vertical metric")
    plt.colorbar()
    # Save plotted graph
    fig2.savefig(os.path.join(prm.outdir, adjmat_filename + ".png"), bbox_inches="tight")
    plt.close(fig2)

    # If image is too big, plotting a graph will be expensive and useless because we won't
    # differentiate the edges.
    if prm.save_graph_plot and n < 50:
        fig3 = plt.figure()
        G = npu.get_2dgrid_weighted_adjacency_npsparse(n, weights)
        # Draw the graph
        import networkx as nx

        g = nx.from_scipy_sparse_matrix(G)
        # The list of edges is in a weird order, so we need to index G with it.
        # Transpose the list of indices because np.array takes (list of x, list of y)
        # This is faster than 'c = np.array(list(g.edges.data(data='weight')))[:, 2]'
        c = np.array(list(g.edges.data(data='weight')))[:, 2]
        cmap = plt.cm.get_cmap('coolwarm')
        positions = dict(zip(np.arange(N), np.vstack(map(np.ravel, np.meshgrid(np.arange(n), n - 1 - np.arange(n)))).T))
        # Draw only edges, no nodes
        nx.draw_networkx(g, pos=positions, with_labels=False, node_size=0, width=4, edge_color=c, edge_cmap=cmap)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.cm.colors.Normalize(vmin=c.min(), vmax=c.max()))
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.cm.colors.LogNorm(vmin=c.min(), vmax=c.max()))
        sm.set_array([])  # The array can't be None, so it needs to be set to something
        plt.colorbar(sm)

        # Save plotted graph
        fig3.savefig(os.path.join(prm.outdir, adjgraph_filename + ".png"), bbox_inches="tight")
        plt.close(fig3)

    # Compute the kernel from the adjacency matrix
    xi_torch, exp_C, _ = mlc.compute_kernel_torch(prm.from_numpy(weights),n,2,apsp_algo,extra_param)

    # Make the kernel operator and matrix usable with Numpy.
    xi = lambda x : xi_torch(prm.from_numpy(x)).numpy()
    exp_C = exp_C.detach().numpy()


if kernel_check:
    # Get the result of a central Dirac
    mlk.check_kernel(xi_torch,n,N,extra_param)


# # Compute displacement interpolation with barycenters
P = np.array([p.flatten(),q.flatten()])

# Plot only the isobarycenter
# plt.ion()
# plt.figure()
# w = np.array([0.5,0.5])
# interp, _, _ = OT_Sinkhorn.compute_sinkhorn_barycenter(P,w,xi,L)[0]
# plt.imshow(np.reshape(interp,[n,n]),cmap='gray')
# plt.show()
# plt.pause(2)
# plt.close()

# Interpolate in the Wasserstein space
# plt.ioff()
# plt.figure()
barys = np.zeros([k,n,n])
for i in range(k):
    alpha = i/(k-1)
    print("alpha = ",alpha)
    w = np.array([1-alpha,alpha])
    interp, err_p, err_q = OT_Sinkhorn.compute_sinkhorn_barycenter(P,w,xi,L)
    # interp, err = OT_Sinkhorn.compute_sinkhorn_barycenter_log(P,w,C,L,gamma)

    if sink_check:
        # Plot Sinkhorn error
        fig = plt.figure()
        plt.semilogy(err_p)
        plt.semilogy(err_q)
        plt.legend(["err_p", "err_q"])
        plt.title("Sinkhorn marginal constraint on p and q")
        fig.savefig(os.path.join(prm.outdir, "sink_conv-%02d.png"%i), bbox_inches="tight")
        plt.close(fig)

    # plt.subplot(1,k,i)
    I = np.reshape(interp, [n,n])
    # plt.imshow(I, cmap='gray')
    barys[i] = I

    beta = 1.0/np.max(I)
    # print(beta)
    imageio.imsave(os.path.join(prm.outdir,"image-%02d.png"%i),(I*beta*255).astype('uint8'))

    np.save(os.path.join(prm.outdir,"array-%02d"%i),I)

    # plt.pause(0.01)
    # plt.show()

# Save images with the same scaling
scale = np.max(barys)
for i in range(k):
    imageio.imsave(os.path.join(prm.outdir, "image-globalscale-%02d.png" % i), (barys[i] / scale*255).astype('uint8'))
