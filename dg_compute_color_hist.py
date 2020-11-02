#!/usr/bin/env python
"""
dg_compute_color_hist.py

Compute the color histograms of images, and save them in .npy files, as well as a 3D plot.
"""

import matplotlib
# print("Matplotlib backend: ", matplotlib.get_backend())
matplotlib.use('agg')
# print("Matplotlib backend: ", matplotlib.get_backend())
import matplotlib.pyplot as plt

import numpy as np
import glob
import imageio
import os
import argparse
import cv2

import ml_color_transfer as mlct

__author__ = "Matthieu Heitz"


def arg_start():

    # Set default parameters
    def_prm = {}
    def_prm['n'] = 16
    def_prm['outdir'] = ""
    def_prm['colorspace'] = "RGB"
    def_prm['scale'] = 1.0

    help_dict = {}
    help_dict['n'] = "Color histogram size (n*n*n)"
    help_dict['outdir'] = "Output directory where data will be saved"
    help_dict['colorspace'] = "Color space : RGB or LAB"
    help_dict['scale'] = "Scale for point size in the 3D scatter plot of the histogram"

    # Add some short arguments
    short_args = {}
    short_args['n'] = 'n'
    short_args['outdir'] = 'o'
    short_args['colorspace'] = 'c'
    short_args['scale'] = 's'

    # Helper function to add 2 mutualy exclusive boolean optional arguments : --flag and --no-flag
    def add_boolean_opt_arg(parser, def_dict, name, help_str):
        if type(def_dict[name]) != bool:
            print("Error: Default value for '%s' should be a boolean.")
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true', help=help_str+". Default value is '"+str(def_dict[name]) + "'")
        group.add_argument('--no-' + name, dest=name, action='store_false', help=help_str+". Default value is '"+str(not def_dict[name]) + "'")
        parser.set_defaults(**{name: def_dict[name]})

    # One-liner that adds an optional argument whose default value is in def_dict
    def add_general_opt_arg(parser, def_dict, name, type, help_str):
        if name in short_args:
            parser.add_argument("-"+short_args[name],"--"+name, default=def_dict[name], type=type,
                                help=help_str+". Default value is '"+str(def_dict[name]) + "'")
        else:
            parser.add_argument("--"+name, default=def_dict[name], type=type,
                                help=help_str+". Default value is '"+str(def_dict[name]) + "'")

    parser = argparse.ArgumentParser(description="GMLG - compute color histograms of images", epilog="")
    parser.add_argument("in_files", type=str, help="Glob pattern matching input files (e.g. '/path/to/files*.png'))")

    for name in def_prm:
        if type(def_prm[name]) != bool:
            add_general_opt_arg(parser, def_prm, name, type=type(def_prm[name]), help_str=help_dict[name])
        else:
            add_boolean_opt_arg(parser, def_prm, name, help_str=help_dict[name])

    # Get all parameters, with values from argument, or from default
    args = parser.parse_args()

    # Update the parameter dict with the passed arguments
    prm_dict = args.__dict__.copy()

    # Get variables needed in this function
    in_files = prm_dict['in_files']
    n = prm_dict['n']
    scale = prm_dict['scale']
    outdir = prm_dict['outdir']
    colorspace = prm_dict['colorspace']

    base_dir = os.path.dirname(in_files)

    # Read interpolations
    files = np.sort(glob.glob(in_files))
    K = len(files)
    if K == 0:
        print("ERROR: Found 0 files that match '%s'"%in_files)
        exit(-1)
    if not os.path.isfile(files[0]):
        print("ERROR: The following pattern matches a directory: '%s'" % in_files)
        exit(-1)

    if outdir == "":
        outdir_name = "hists%d" % n
        if colorspace == "LAB": outdir_name += "lab"
        outdir = os.path.join(base_dir, outdir_name)

    print("Input dir: %s"%in_files)
    print("Output dir: %s"%outdir)

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    scales = np.zeros(K)
    vmin = 1e-6
    N = n**3
    normalize = lambda p: p / np.sum(p)
    print()
    file_hist_npy = "hist-array"
    file_hist_png = "hist-image"
    if colorspace == "LAB":
        file_hist_npy += "-lab"
        file_hist_png += "-lab"

    for i in range(K):
        print("Processing %s"%files[i])
        I = np.array(imageio.imread(files[i]))
        if colorspace == "LAB":
            If32 = I.astype(np.float32)/255
            If32_lab = cv2.cvtColor(If32,cv2.COLOR_RGB2LAB)
            If32_lab[:,:,0] *= 255/100; If32_lab[:,:,1:] = If32_lab[:,:,1:] + 128
            I = If32_lab
        hist_item = cv2.calcHist([I], [0,1,2], None, [n,n,n], [0, 255, 0, 255, 0, 255])
        scales[i] = np.sum(hist_item)
        hist_item = normalize(hist_item + vmin*np.max(hist_item)/N)
        mlct.save_color_hist(hist_item,os.path.join(outdir,file_hist_png+"-%03d"%i),scale,save_npy=False,colorspace=colorspace)
        np.save(os.path.join(outdir,file_hist_npy+"-%03d"%i),hist_item)

    print("Done")
    return 0


if __name__ == "__main__":

    arg_start()

