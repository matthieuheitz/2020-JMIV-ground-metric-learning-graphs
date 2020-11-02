#!/usr/bin/env python
"""
ml_direct_transfer.py

Script to directly transfer colors of an image onto another, without OT interpolation, or learned metric stuff.

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
import ml_color_timelapse
import OT_Sinkhorn
import ml_color_transfer as mlct

# For MacOS
matplotlib.use("agg")   # Non-interactive backend
# matplotlib.use("Qt4Agg") # Interactive backend
# plt.get_current_fig_manager().window.wm_geometry("+1600+400") # For TkAgg
# plt.get_current_fig_manager().window.setGeometry(1600, 400, 1000, 800) # For QtAgg

__author__ = "Matthieu Heitz"


def direct_transfer():

    def_prm = {}
    def_prm['top_outdir'] = "."
    def_prm['final_outdir'] = ""
    def_prm['omp_num_threads'] = ""
    # Parameters for color transfer
    def_prm['colorspace'] = "RGB"   # RGB, LAB
    def_prm['n'] = 32
    def_prm['L'] = 100
    def_prm['sig_gamma'] = 0.0
    def_prm['gamma'] = 0.0
    def_prm['sink_check'] = True
    def_prm['kernel_check'] = False
    def_prm['apply_bilateral_filter'] = True
    def_prm['bf_sigmaSpatial'] = 0.0
    def_prm['bf_sigmaRange'] = 0.0
    def_prm['combinations'] = "all"     # all, match

    help_dict = {}
    help_dict['top_outdir'] = "Top directory for output, the result will go in a directory named with parameters, in that top directory."
    help_dict['final_outdir'] = "Override 'top_outdir' and writes directly in final_directory"
    help_dict['omp_num_threads'] = "Set the environment variable OMP_NUM_THREADS"
    # Parameters for color transfer
    help_dict['colorspace'] = "Colorspace in which to perform the transfer: {RGB,LAB}. Default is RGB"
    help_dict['L'] = "Number of Sinkhorn iterations for color transfer. When 0, use the value in the metric params"
    help_dict['n'] = "Size of color histograms (n**3). Default is 32"
    help_dict['sig_gamma'] = "Sigma value for the gaussian convolution (for hist_interp='euclid') for color transfer"
    help_dict['gamma'] = "Gamma value for the gaussian convolution (for hist_interp='euclid') for color transfer. gamma=2*sigma**2"
    help_dict['sink_check'] = "Whether to save the plot of Sinkhorn errors for the color transfer"
    help_dict['kernel_check'] = "Check the kernel by diffusing a central Dirac"
    help_dict['apply_bilateral_filter'] = "Whether to apply a cross bilateral filter after color transfer"
    help_dict['bf_sigmaSpatial'] = "Sigma for the spatial gaussian in the cross bilateral filter. Set to 0 for default"
    help_dict['bf_sigmaRange'] = "Sigma for the image gaussian in the cross bilateral filter. Set to 0 for default"
    help_dict['combinations'] = "Combinations between src and target images: all (each src on each tgt), match (each src to its corresponding tgt)"

    # Add some short arguments
    short_args = {}
    short_args['top_outdir'] = 'o'
    short_args['n'] = 'n'
    short_args['L'] = 'L'

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

    parser = argparse.ArgumentParser(description="GMLG - Do a direct color transfer of source image colors on target images", epilog="")
    parser.add_argument("source_images", type=str, help="Glob pattern matching source images (e.g. \"/path/to/image*.png\")")
    parser.add_argument("target_images", type=str, help="Glob pattern matching target images (e.g. \"/path/to/image*.png\")")

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

    if 'sig_gamma' in passed_prm and 'gamma' in passed_prm:   print("Can't pass both 'sig_gamma' and 'gamma'."); exit()
    elif 'sig_gamma' in passed_prm:    args_dict['gamma'] = 2*args_dict['sig_gamma']**2
    elif 'gamma' in passed_prm:        args_dict['sig_gamma'] = np.sqrt(args_dict['gamma']/2)
    else: print("Must specify either 'sig_gamma' or 'gamma', and not both."); exit()

    # Positional arguments
    source_images = args_dict['source_images']
    target_images = args_dict['target_images']
    n = args_dict['n']
    # CT arguments
    colorspace = args_dict['colorspace']
    L = args_dict['L']
    sig_gamma = args_dict['sig_gamma']
    gamma = args_dict['gamma']
    sink_check = args_dict['sink_check']
    kernel_check = args_dict['kernel_check']
    apply_bilateral_filter = args_dict['apply_bilateral_filter']
    bf_sigmaSpatial = args_dict['bf_sigmaSpatial']
    bf_sigmaRange = args_dict['bf_sigmaRange']
    # Other arguments
    top_outdir = args_dict['top_outdir']
    final_outdir = args_dict['final_outdir']
    combinations = args_dict['combinations']


    # Define non-argument variables.
    dim = 3
    N = n**dim
    print("Histogram size for color transfer:",n)
    source_dataset = source_images.split('/')[-2]
    target_dataset = target_images.split('/')[-2]

    # Read images
    source_files = np.sort(glob.glob(source_images))
    target_files = np.sort(glob.glob(target_images))
    if not source_files.size:
        print("ERROR: No files are matched by '%s'."%source_images)
        exit(-1)
    if not target_files.size:
        print("ERROR: No files are matched by '%s'."%target_images)
        exit(-1)
    for f in source_files:
        if not os.path.isfile(f):
            print("ERROR: The glob pattern for source images should only match files, not directories.")
            exit(-1)
    for f in target_files:
        if not os.path.isfile(f):
            print("ERROR: The glob pattern for target images should only match files, not directories.")
            exit(-1)
    num_src_hist = len(source_files)
    num_tgt_hist = len(target_files)

    # Output dir
    # Two modes:
    # 1: specify final_outdir, and data will just go there.
    if final_outdir:
        prm.outdir = final_outdir
    # 2: final_outdir is empty and the program generates a name with parameters.
    else:
        # If we use an existing interp, take the same folder name and append the parameters for color transfer
        prm.outdir = os.path.join(top_outdir,"test_dt")
        prm.outdir += "_" + source_dataset + "-" + str(num_src_hist) + "_" + target_dataset + "-" + str(num_tgt_hist)
        prm.outdir += "_nct%d_Lct%d_gct%0.2g"%(n,L,gamma)

        if apply_bilateral_filter:
            prm.outdir += "__bf_sigS%0.2g_sigR%0.2g"%(bf_sigmaSpatial,bf_sigmaRange)

        if combinations == "match":
            prm.outdir += "_match"

    os.makedirs(prm.outdir, exist_ok=True)

    # Compute kernel
    # xi = mlc.compute_3d_euclidean_kernel_numpy(n, "convolution_npy", gamma)
    xi = mlc.compute_3d_euclidean_kernel_native(n, gamma)

    # Check kernel
    if kernel_check:
        metric_prm = {}
        metric_prm['t_heat'] = gamma    # Make check_kernel think it's a heat-based kernel.
        ml_color_timelapse.check_kernel(xi,n,N,metric_prm,kernel_io_torch=False)

    if combinations == "match" and num_src_hist != num_tgt_hist:
        print("ERROR: If combinations=='match', there should be as many sources than targets.")
        exit(-1)

    vmin = 1e-6  # Add minimal mass to histograms (later divided by N)
    normalize = lambda p: p / np.sum(p)

    # Read source images
    src_images = {}     # Allows input images of different sizes
    src_images_rgb = {}     # Save RGB versions of input images
    src_hists = np.zeros([num_src_hist, N])
    src_scales = np.zeros(num_src_hist)
    for i in np.arange(0, num_src_hist):     # num_src_hist is generally 2: Beginning and enf of the interpolation
        src_images[i] = np.array(imageio.imread(source_files[i]))
        src_images_rgb[i] = src_images[i]
        if colorspace == "LAB":
            If32 = src_images[i].astype(np.float32)/255
            If32_lab = cv2.cvtColor(If32,cv2.COLOR_RGB2LAB)
            If32_lab[:,:,0] *= 255/100; If32_lab[:,:,1:] = If32_lab[:,:,1:] + 128
            src_images[i] = If32_lab
        hist_item = cv2.calcHist([src_images[i]], [0, 1, 2], None, [n, n, n], [0, 255, 0, 255, 0, 255])
        src_scales[i] = np.sum(hist_item)
        hist_item = normalize(hist_item + vmin*np.max(hist_item)/N)
        # save_color_hist(hist_item, os.path.join(prm.outdir, "input-hist-%02d" % i), 1) # Save input histograms
        src_hists[i] = hist_item.flatten()

    # Read target images
    tgt_images = {}     # Allows input images of different sizes
    tgt_images_rgb = {}     # Save RGB versions of input images
    tgt_hists = np.zeros([num_tgt_hist, N])
    tgt_scales = np.zeros(num_tgt_hist)
    for i in np.arange(0, num_tgt_hist):     # num_tgt_hist is generally 2: Beginning and enf of the interpolation
        tgt_images[i] = np.array(imageio.imread(target_files[i]))
        tgt_images_rgb[i] = tgt_images[i]
        if colorspace == "LAB":
            If32 = tgt_images[i].astype(np.float32)/255
            If32_lab = cv2.cvtColor(If32,cv2.COLOR_RGB2LAB)
            If32_lab[:,:,0] *= 255/100; If32_lab[:,:,1:] = If32_lab[:,:,1:] + 128
            tgt_images[i] = If32_lab
        hist_item = cv2.calcHist([tgt_images[i]], [0, 1, 2], None, [n, n, n], [0, 255, 0, 255, 0, 255])
        tgt_scales[i] = np.sum(hist_item)
        hist_item = normalize(hist_item + vmin*np.max(hist_item)/N)
        # save_color_hist(hist_item, os.path.join(prm.outdir, "input-hist-%02d" % i), 1) # Save input histograms
        tgt_hists[i] = hist_item.flatten()

    # Create parameter dictionary
    param_dict = {
        'source_images': source_images,
        'target_images': target_images,

        'n': n,
        'L': L,
        'sig_gamma': sig_gamma,
        'apply_bilateral_filter': apply_bilateral_filter,
        'bf_sigmaSpatial': bf_sigmaSpatial,
        'bf_sigmaRange': bf_sigmaRange,

        'outdir': prm.outdir,
        'num_src_hist': num_src_hist,
        'vmin': vmin,
        'N': N,
        'gamma': gamma,
        'colorspace': colorspace,
    }
    # Save parameters
    out_param_file = os.path.join(prm.outdir, '0-parameters.json')
    npu.write_dict_to_json(out_param_file, param_dict)

    # Save histograms
    src_hist_filename = "src-hist"
    tgt_hist_filename = "tgt-hist"
    if colorspace == "LAB":
        src_hist_filename += "-lab"
        tgt_hist_filename += "-lab"
    for i in range(num_src_hist):
        mlct.save_color_hist(src_hists[i].reshape([n,n,n]), os.path.join(prm.outdir, src_hist_filename+"-image-%02d-n%d"%(i,n)), 1, colorspace=colorspace, save_npy=False)
        np.save(os.path.join(prm.outdir, src_hist_filename +"-array-%02d-n%d"%(i,n)), src_hists[i].reshape([n,n,n]))
    for i in range(num_tgt_hist):
        mlct.save_color_hist(tgt_hists[i].reshape([n,n,n]), os.path.join(prm.outdir, tgt_hist_filename+"-image-%02d-n%d"%(i,n)), 1, colorspace=colorspace, save_npy=False)
        np.save(os.path.join(prm.outdir, tgt_hist_filename +"-array-%02d-n%d"%(i,n)), tgt_hists[i].reshape([n,n,n]))

    # Transfer each source on each target
    for i in np.arange(num_tgt_hist):
        for j in range(num_src_hist):

            if combinations == "match" and i != j:
                continue

            interp_hist_array_filename = "interp-hist-array"
            interp_hist_image_filename = "interp-hist-image"
            if colorspace == "LAB":
                interp_hist_array_filename += "-lab"
                interp_hist_image_filename += "-lab"

            print("Transferring colors of %s on %s"%(os.path.basename(source_files[j]),os.path.basename(target_files[i])))
            sink_conv_file = "sink_conv-%02d-%02d" % (i,j) if sink_check else ""
            im = mlct.color_transfer_ot_barycentric_proj(tgt_images[i]/255.0, tgt_hists[i], src_hists[j], xi, L, sink_conv_filebase=sink_conv_file,
                                                         apply_bilateral_filter=apply_bilateral_filter, sigmaRange=bf_sigmaRange,
                                                         sigmaSpatial=bf_sigmaSpatial, colorspace=colorspace)

            # Save image before bilteral filtering
            if colorspace == "LAB":  # Convert to RGB before saving image
                # C = (np.vstack((x.flatten(), y.flatten(), z.flatten())).T).astype(np.float32)[T.flatten()]
                im_lab = im.copy().astype(np.float32)
                im_lab[:, :, 0] *= 100; im_lab[:, :, 1:] = im_lab[:, :, 1:] * 255 - 128  # Put in the good range
                im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)

            I = Image.fromarray((im * 255).astype('uint8'), 'RGB')
            I.save(os.path.join(prm.outdir, "interp-%02d-%02d.png" % (i,j)))

    print("Done")




if __name__ == "__main__":

    direct_transfer()
