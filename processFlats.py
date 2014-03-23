# DM processing of Phosim output
# You should 'setup pipe_test' to use it.
# C. Walter 01/2014

import math
import numpy                as np

import lsst.afw.math        as afwMath
import lsst.afw.table       as afwTable
import lsst.afw.image       as afwImg
import lsst.afw.detection   as afwDetect

import lsst.meas.algorithms as measAlg

statFlags = (afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV | afwMath.MAX | 
             afwMath.MIN | afwMath.ERRORS)
print "The statistics flags are set to %s."%bin(statFlags)
print "Errors will be calculated.\n"

outDir       = 'output/lsst_flats_e_'
suffix       = '_f2_R22_S11_E000.fits.gz'

numElectrons  = '14'
extraId = '2'

fileName    = outDir+numElectrons+extraId+suffix
print "Processing file ", fileName

exposure    = afwImg.ExposureF(fileName)
maskedImage = exposure.getMaskedImage()

# These three are held in the maskedImage
image       = maskedImage.getImage()
mask        = maskedImage.getMask()
variance    = maskedImage.getVariance()

imageStatistics = afwMath.makeStatistics(maskedImage, statFlags)
numBins         = imageStatistics.getResult(afwMath.NPOINT)[0]
mean            = imageStatistics.getResult(afwMath.MEAN)[0]

print "The image has dimensions %i x %i pixels" \
    %(maskedImage.getWidth(), maskedImage.getHeight())
print "Number of analyzed bins in image is %i"  %numBins
print "Max    = %9d"            %imageStatistics.getResult(afwMath.MAX)[0]
print "Min    = %9d"            %imageStatistics.getResult(afwMath.MIN)[0]
print "Mean   = %9.3f +- %3.3f" %imageStatistics.getResult(afwMath.MEAN)
print "StdDev = %9.2f"          %imageStatistics.getResult(afwMath.STDEV)[0]

def submatrix(M,i,j):
        return M[i-1:i+2,j-1:j+2].ravel()

a = image.getArray().T

print "\nCalculate 2D spatial Autocorrelation"
print "shape is", a.shape

np.set_printoptions(precision=3)

# Take a 3x3 matrix around each pixel and then contruct a matrix for which
# the row is one of those pixels labels  and the column is the value for this
# pixel.  Repeat this for each pixel.  The result is a 3x3=9 rows and 
# N columns (one for each pixel). Ignore the outer edges since they don't
# have the the full set of neighbors.

q = [ submatrix(a,i,j) for i,j in np.ndindex(a.shape) 
      if i != 0 and j != 0 and i != a.shape[0]-1 and j != a.shape[1]-1]

y = np.column_stack(q)

fullCorrellation        = np.corrcoef(y)
centerPixelCorrellation = fullCorrellation[5-1].reshape(3,3)

print centerPixelCorrellation
