#!/usr/bin/env python
"""
dg_kinterp2_from_images.py

Data Generation for the application ml_kinterp2
This script generates the histograms and marginals in .npy format, from a sequence of image files.
This is useful to always keep the input data in the same format, for the ml_kinterp2 algorithm.
Moreover, since we save the scale of each image before transforming it to a histogram, it can later
be read from the parameter file, and used to output the reconstructions to the good scale.

This script doesn't fix the problem that if images don't have the same mass, the ones with less mass will have a higher
max value than those with more mass, because we divide each image by its total sum of values so that it sums to 1.

The fact of rescaling images with their initial sum in order to visualize them doesn't change the fact that when they
are in the form of histograms, the problem of oscillating levels is still there: the same region with the same original
color in every frame will change value from one image to another, because they each have been normalized by their sum.
In the end, if we don't add the missing mass somewhere in each image, sensefully according to the data, then it doesn't
really matter how we render the images to png, because it doesn't change how the OT algorithm sees them.
Perhaps it is wise to use the same rescaling value for all (the max value of the histograms), so that we see that levels
change in the images, so we are reminded that internally, that is how the algorithm sees them.

"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from scipy import sparse
import math
import time
import sys
import glob

import OT_Sinkhorn

from ml_parameters import *
import np_utils as npu
import ml_core as mlc


__author__ = "Matthieu Heitz"

# Input
# input_files = "data/toybubble/image-*.png"
input_files = "/path/to/input/folder/image-*.png"

# Initialize parameters
outdir = os.path.dirname(input_files)
vmin = 1e-6 # Minimal mass (as a ratio of max value)

# Read image files
files = np.sort(glob.glob(input_files))
if len(files) == 0:
    print("ERROR: Found 0 files that match '%s'"%input_files)
    exit(-1)
K = len(files)
s = imageio.imread(files[0]).shape
channels = 0
if len(s) == 2:
    nrow, ncol = imageio.imread(files[0]).shape
    channels = 1
elif len(s) == 3:
    nrow, ncol = imageio.imread(files[0]).shape[:-1]
    channels = s[-1]
N = ncol*nrow
obs = np.zeros([K,nrow,ncol])
scales = np.zeros(K)
normalize = lambda p: p/np.sum(p)
for i in range(K):
    im = np.array(imageio.imread(files[i])) # "PNG-PIL" is default
    # im = np.array(imageio.imread(files[i],"PNG-FI"))
    if channels > 1:
        # Convert RGB to gray : average channels equally, but ignore alpha channel if present
        im = np.mean(im[:, :, :3], 2)

    scales[i] = np.sum(im)
    # Add minimal mass
    obs[i] = normalize(im + np.max(im)*vmin)
    # Save in NPY format
    # np.save(os.path.join(outdir, "array-%02d"%i), obs[i])

# Save images of histograms, normalized by their global maximum
# This gives images that render better how the histograms really are.
# If we normalize each one by its maximum, we get sequences that "flicker"
# If we normalize each one by its original sum of pixels, we get sequences that don't "flicker",
# but it makes us forget that the algorithm sees them flickering.
obs_max = np.max(obs)
for i in range(K):
    # obs[i] = normalize(obs[i])
    imageio.imsave(os.path.join(outdir, 'normalized-image-%02d.png'%i), (obs[i]/obs_max*255).astype('uint8'))
    # Save in NPY format
    np.save(os.path.join(outdir, "array-%02d"%i), obs[i])

# Marginals are simply the first and last frames
p = obs[0]
q = obs[K-1]

# Save parameters
param_dict = {}
param_dict['nrow'] = nrow
param_dict['ncol'] = ncol
param_dict['N'] = N
param_dict['vmin'] = vmin
param_dict['K'] = K
param_dict['scale_mean'] = np.mean(scales)
for i in range(K):
    param_dict['scale-%02d'%i] = scales[i]

param_file_fp = os.path.join(outdir, '0-parameters.json')
npu.write_dict_to_json(param_file_fp,param_dict)

# # Show input histograms
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
# # plt.show()