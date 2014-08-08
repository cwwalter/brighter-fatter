# DM processing of Phosim output
# Subtract two flats and calculate statistics
# You should 'setup pipe_test' to use it.
# C. Walter 01/2014

import math                 as math
import numpy                as np

import lsst.afw.math        as afwMath
import lsst.afw.table       as afwTable
import lsst.afw.image       as afwImg
import lsst.afw.detection   as afwDetect

def submatrix(M,i,j):
    return M[i-1:i+2,j-1:j+2].ravel()

def processImage(maskedImage):

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

    a = image.getArray().T

    print "\nCalculate 2D spatial Autocorrelation"
    print "shape is", a.shape

    np.set_printoptions(precision=3, suppress=True)

    # Take a 3x3 matrix around each pixel and then contruct a matrix for which
    # the row is one of those pixels labels  and the column is the value for this
    # pixel.  Repeat this for each pixel.  The result is a 3x3=9 rows and 
    # N columns (one for each pixel). Ignore the outer edges since they don't
    # have the the full set of neighbors.

    q = [ submatrix(a,i,j) for i,j in np.ndindex(a.shape) 
        if i != 0 and j != 0 and i != a.shape[0]-1 and j != a.shape[1]-1]

    y = np.column_stack(q)

    fullCorrelation        = np.corrcoef(y)
    centerPixelCorrelation = fullCorrelation[5-1].reshape(3,3)

    print centerPixelCorrelation

    # Now make a new 100x104 matrix to check the mean and stddev.
    # (group each pixel into 4x4 blocks)
    # I barely understand how this works (CWW)!
    rows, cols = a.shape
    b = a.reshape(rows//4,4,cols//4,4).sum(axis=(1, 3))

    print "Original Mean:", np.mean(a), "Std:", np.std(a)
    print "4x4      Mean:", np.mean(b), "Std:", np.std(b)

    horizCorrelation = (centerPixelCorrelation[1,0] + centerPixelCorrelation[1,2])/2.0
    vertCorrelation  = (centerPixelCorrelation[0,1] + centerPixelCorrelation[2,1])/2.0
    return (np.mean(a), np.std(a), np.mean(b), np.std(b), horizCorrelation, vertCorrelation)
        
# Setup global statistics and filenames    
statFlags = (afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV | afwMath.MAX | 
afwMath.MIN | afwMath.ERRORS)
print "The statistics flags are set to %s."%bin(statFlags)
print "Errors will be calculated.\n"

outDir       = 'output/lsst_flats_e_'
suffix       = '_f2_R22_S11_E000.fits.gz'

# Process Files
numElectrons   = '12'
extraId = '0'

numElectrons1  = numElectrons+'0'
numElectrons2  = numElectrons+'1'
fileName1 = outDir+numElectrons1+extraId+suffix
fileName2 = outDir+numElectrons2+extraId+suffix

# Get images
maskedImage1 = afwImg.ExposureF(fileName1).getMaskedImage()
maskedImage2 = afwImg.ExposureF(fileName2).getMaskedImage()
maskedImage3 = maskedImage1.clone()
maskedImage3 -= maskedImage2

# Process images
print "Processing file ", fileName1
(mean1, std1, groupMean1, groupStd1, hCorr1, vCorr1) = processImage(maskedImage1)
print "\nProcessing file ", fileName2
(mean2, std2, groupMean2, groupStd2, hCorr2, vCorr2) = processImage(maskedImage2)
print "\nProcessing Difference"
(mean3, std3, groupMean3, groupStd3, hCorr3, vCorr3) = processImage(maskedImage3)

# Print results
print "\n---Results for magnitude", numElectrons, "config", extraId,":\n"

print "Image1:\t\t %9.2f %9.2f %7.2f   "% (mean1, std1, std1**2/mean)
print "Image3:\t\t %9.2f %9.2f %7.2f \n"% (mean3, std3, (std3/math.sqrt(2))**2/mean1)

print "Grouped1:\t %9.2f %9.2f %7.2f   "% (groupMean1, groupStd1, groupStd1**2/groupMean1)
print "Grouped3:\t %9.2f %9.2f       \n"% (groupMean3, groupStd3)

print "Correlation1:\t %9.3f %9.3f"% (hCorr1, vCorr1)
print "Correlation3:\t %9.3f %9.3f"% (hCorr3, vCorr3)
