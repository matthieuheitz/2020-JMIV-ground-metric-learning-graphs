"""
bilateral_approximation.py
Fast Bilateral Filter Approximation Using a Signal Processing Approach in Python

Copyright (c) 2014 Jack Doerner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Doc taken from the matlab version from which this was inspired
# http://people.csail.mit.edu/jiawen/software/bilateralFilter.m

# Bilateral and Cross-Bilateral Filter using the Bilateral Grid.
#
# Bilaterally filters the image 'data' using the edges in the image 'edge'.
# If 'data' == 'edge', then it the standard bilateral filter.
# Otherwise, it is the 'cross' or 'joint' bilateral filter.
# For convenience, you can also pass in [] for 'edge' for the normal
# bilateral filter.
#
# Note that for the cross bilateral filter, data does not need to be
# defined everywhere.  Undefined values can be set to 'NaN'.  However, edge
# *does* need to be defined everywhere.
#
# data and edge should be of the greyscale, double-precision floating point
# matrices of the same size (i.e. they should be [ height x width ])
#
# data is the only required argument
#
# edgeMin and edgeMax specifies the min and max values of 'edge' (or 'data'
# for the normal bilateral filter) and is useful when the input is in a
# range that's not between 0 and 1.  For instance, if you are filtering the
# L channel of an image that ranges between 0 and 100, set edgeMin to 0 and
# edgeMax to 100.
#
# edgeMin defaults to min( edge( : ) ) and edgeMax defaults to max( edge( : ) ).
# This is probably *not* what you want, since the input may not span the
# entire range.
#
# sigmaSpatial and sigmaRange specifies the standard deviation of the space
# and range gaussians, respectively.
# sigmaSpatial defaults to min( width, height ) / 16
# sigmaRange defaults to ( edgeMax - edgeMin ) / 10.
#
# samplingSpatial and samplingRange specifies the amount of downsampling
# used for the approximation.  Higher values use less memory but are also
# less accurate.  The default and recommended values are:
#
# samplingSpatial = sigmaSpatial
# samplingRange = sigmaRange

import numpy
import math
import scipy.signal, scipy.interpolate


# Applies the bilateral filter on a color image, by applying it separately on each channel
def bilateral_approximation_color(data, edge, sigmaS, sigmaR, samplingS=None, samplingR=None, edgeMin=None, edgeMax=None):
	if len(data.shape) != 3 or data.shape[2] != 3:
		print("ERROR: This function only accepts inputs of shape [h,w,3]")
		exit(-1)
	out0 = bilateral_approximation(data[:, :, 0], edge[:, :, 0], sigmaS, sigmaR, samplingS=samplingS, samplingR=samplingR, edgeMin=edgeMin, edgeMax=edgeMax)
	out1 = bilateral_approximation(data[:, :, 1], edge[:, :, 1], sigmaS, sigmaR, samplingS=samplingS, samplingR=samplingR, edgeMin=edgeMin, edgeMax=edgeMax)
	out2 = bilateral_approximation(data[:, :, 2], edge[:, :, 2], sigmaS, sigmaR, samplingS=samplingS, samplingR=samplingR, edgeMin=edgeMin, edgeMax=edgeMax)
	return numpy.stack((out0, out1, out2), 2)


# Applies the bilateral filter on a 1-channel image
def bilateral_approximation(data, edge, sigmaS, sigmaR, samplingS=None, samplingR=None, edgeMin=None, edgeMax=None):
	# This function implements Durand and Dorsey's Signal Processing Bilateral Filter Approximation (2006)
	# It is derived from Jiawen Chen's matlab implementation
	# The original papers and matlab code are available at http://people.csail.mit.edu/sparis/bf/

	inputHeight = data.shape[0]
	inputWidth = data.shape[1]
	samplingS = sigmaS if (samplingS is None) else samplingS
	samplingR = sigmaR if (samplingR is None) else samplingR
	edgeMax = numpy.amax(edge) if (edgeMax is None) else edgeMax
	edgeMin = numpy.amin(edge) if (edgeMin is None) else edgeMin
	edgeDelta = edgeMax - edgeMin
	derivedSigmaS = sigmaS / samplingS
	derivedSigmaR = sigmaR / samplingR

	paddingXY = math.floor( 2 * derivedSigmaS ) + 1
	paddingZ = math.floor( 2 * derivedSigmaR ) + 1

	# allocate 3D grid
	downsampledWidth = math.floor( ( inputWidth - 1 ) / samplingS ) + 1 + 2 * paddingXY
	downsampledHeight = math.floor( ( inputHeight - 1 ) / samplingS ) + 1 + 2 * paddingXY
	downsampledDepth = math.floor( edgeDelta / samplingR ) + 1 + 2 * paddingZ

	gridData = numpy.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )
	gridWeights = numpy.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )

	# compute downsampled indices
	(jj, ii) = numpy.meshgrid( range(inputWidth), range(inputHeight) )

	di = numpy.around( ii / samplingS ).astype('int') + paddingXY
	dj = numpy.around( jj / samplingS ).astype('int') + paddingXY
	dz = numpy.around( ( edge - edgeMin ) / samplingR ).astype('int') + paddingZ

	# perform scatter (there's probably a faster way than this)
	# normally would do downsampledWeights( di, dj, dk ) = 1, but we have to
	# perform a summation to do box downsampling
	for k in range(dz.size):
	
		dataZ = data.flat[k]
		if (not math.isnan( dataZ  )):
			
			dik = di.flat[k]
			djk = dj.flat[k]
			dzk = dz.flat[k]

			gridData[ dik, djk, dzk ] += dataZ
			gridWeights[ dik, djk, dzk ] += 1

	# make gaussian kernel
	kernelWidth = 2 * derivedSigmaS + 1
	kernelHeight = kernelWidth
	kernelDepth = 2 * derivedSigmaR + 1
	
	halfKernelWidth = math.floor( kernelWidth / 2 )
	halfKernelHeight = math.floor( kernelHeight / 2 )
	halfKernelDepth = math.floor( kernelDepth / 2 )

	(gridX, gridY, gridZ) = numpy.meshgrid( range( int(kernelWidth) ), range( int(kernelHeight) ), range( int(kernelDepth) ) )
	gridX -= halfKernelWidth
	gridY -= halfKernelHeight
	gridZ -= halfKernelDepth
	gridRSquared = (( gridX * gridX + gridY * gridY ) / ( derivedSigmaS * derivedSigmaS )) + (( gridZ * gridZ ) / ( derivedSigmaR * derivedSigmaR ))
	kernel = numpy.exp( -0.5 * gridRSquared )
	
	# convolve
	blurredGridData = scipy.signal.fftconvolve( gridData, kernel, mode='same' )
	blurredGridWeights = scipy.signal.fftconvolve( gridWeights, kernel, mode='same' )

	# divide
	blurredGridWeights = numpy.where( blurredGridWeights == 0 , -2, blurredGridWeights) # avoid divide by 0, won't read there anyway
	normalizedBlurredGrid = blurredGridData / blurredGridWeights
	normalizedBlurredGrid = numpy.where( blurredGridWeights < -1, 0, normalizedBlurredGrid ) # put 0s where it's undefined

	# upsample
	( jj, ii ) = numpy.meshgrid( range( inputWidth ), range( inputHeight ) )
	# no rounding
	di = ( ii / samplingS ) + paddingXY
	dj = ( jj / samplingS ) + paddingXY
	dz = ( edge - edgeMin ) / samplingR + paddingZ 

	return scipy.interpolate.interpn( (range(normalizedBlurredGrid.shape[0]),range(normalizedBlurredGrid.shape[1]),range(normalizedBlurredGrid.shape[2])), normalizedBlurredGrid, (di, dj, dz) )



def test_bilateral_approximation():

	import matplotlib.pyplot as plt		# Import before matlab.engine
	import matlab.engine
	import imageio
	import numpy as np
	import time

	# Test image : https://imagemagick.org/image/wizard.png
	image_path = "wizard.png"

	A = np.array(imageio.imread(image_path)).astype('double')/255.0
	edge_min = 0.0
	edge_max = 1.0
	sigmaSpatial = float(np.min(A[:,:,0].shape)/16.0)	# Have to be of type 'float'
	sigmaRange = (edge_max-edge_min)/10.0				# Have to be of type 'float'
	# sigmaSpatial = 8.0		# Have to be of type 'float'
	# sigmaRange = 0.1		# Have to be of type 'float'
	print("sigmaSpatial =",sigmaSpatial)
	print("sigmaRange =",sigmaRange)

	# With Python
	print()
	t0 = time.time()
	pA0o = bilateral_approximation(A[:,:,0], A[:,:,0], sigmaSpatial, sigmaRange, edgeMin=edge_min, edgeMax=edge_max)
	pA1o = bilateral_approximation(A[:,:,1], A[:,:,1], sigmaSpatial, sigmaRange, edgeMin=edge_min, edgeMax=edge_max)
	pA2o = bilateral_approximation(A[:,:,2], A[:,:,2], sigmaSpatial, sigmaRange, edgeMin=edge_min, edgeMax=edge_max)
	P = np.stack((pA0o, pA1o, pA2o), 2)
	print("Total Time Python:",time.time()-t0)

	# With Matlab
	# Code can be found at http://people.csail.mit.edu/jiawen/software/bilateralFilter.m
	print()
	t0 = time.time()
	eng = matlab.engine.start_matlab()
	print("Matlab engine started, time:",time.time()-t0)
	t1 = time.time()
	mA0 = matlab.double(A[:,:,0].tolist())
	mA1 = matlab.double(A[:,:,1].tolist())
	mA2 = matlab.double(A[:,:,2].tolist())
	print("Time conv data P2M:",time.time()-t1)
	t2 = time.time()
	mA0l = eng.bilateralFilter(mA0,[], edge_min, edge_max, sigmaSpatial, sigmaRange)
	mA1l = eng.bilateralFilter(mA1,[], edge_min, edge_max, sigmaSpatial, sigmaRange)
	mA2l = eng.bilateralFilter(mA2,[], edge_min, edge_max, sigmaSpatial, sigmaRange)
	print("Time bilateral filter:",time.time()-t2)
	t3 = time.time()
	mA0o = np.array(mA0l)
	mA1o = np.array(mA1l)
	mA2o = np.array(mA2l)
	print("Time conv data M2P:",time.time()-t3)
	M = np.stack((mA0o, mA1o, mA2o), 2)
	print("Total Time Matlab:",time.time()-t0)

	print("\nMax error Matlab/Python:",np.max(np.abs(M-P)))
	# Clip eventual numerical precision
	P = np.clip(P,0,1)
	M = np.clip(M,0,1)

	plt.figure()
	plt.subplot(131)
	plt.imshow(A)
	plt.title("Original")
	plt.subplot(132)
	plt.imshow(P)
	plt.title("Python bilateral")
	plt.subplot(133)
	plt.imshow(M)
	plt.title("Matlab bilateral")
	plt.suptitle("sig_r = %f, sig_s = %f"%(sigmaRange,sigmaSpatial))
	plt.show()


if __name__ == "__main__":

	test_bilateral_approximation()
