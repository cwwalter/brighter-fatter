# DM processing of Phosim output
# Subtract two flats and calculate statistics
# You should 'setup pipe_test' to use it.
# C. Walter 01/2014

from collections import defaultdict

import math                 as math
import numpy                as np
import itertools

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

    if printLevel >= 2:    
        print "The image has dimensions %i x %i pixels" \
            %(maskedImage.getWidth(), maskedImage.getHeight())
        print "Number of analyzed bins in image is %i"  %numBins
        print "Max    = %9d"            %imageStatistics.getResult(afwMath.MAX)[0]
        print "Min    = %9d"            %imageStatistics.getResult(afwMath.MIN)[0]
        print "Mean   = %9.3f +- %3.3f" %imageStatistics.getResult(afwMath.MEAN)
        print "StdDev = %9.2f"          %imageStatistics.getResult(afwMath.STDEV)[0]

    a = image.getArray().T

    # We want to remove the edges because of the way the charge sharing is working now.
    # Trim 20 pixels from outside
    trim = 20
    a = a[trim:-trim,trim:-trim]
    
    if printLevel >= 2:        
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

    if printLevel >= 2:    
        print centerPixelCorrelation

    # Now make a new 100x104 matrix to check the mean and stddev.
    # (group each pixel into 4x4 blocks)
    # I barely understand how this works (CWW)!
    rows, cols = a.shape
    b = a.reshape(rows//4,4,cols//4,4).sum(axis=(1, 3))
        
    if printLevel >= 2:    
        print "Original Mean:", np.mean(a), "Std:", np.std(a)
        print "4x4      Mean:", np.mean(b), "Std:", np.std(b)
        print 

    horizCorrelation = (centerPixelCorrelation[1,0] + centerPixelCorrelation[1,2])/2.0
    vertCorrelation  = (centerPixelCorrelation[0,1] + centerPixelCorrelation[2,1])/2.0
    return (np.mean(a), np.std(a), np.mean(b), np.std(b), horizCorrelation, vertCorrelation)

# Main Program
def main():

    global statFlags
    global printLevel

    printLevel = 0

    global mean
    global PTC
    global groupPTC 
    global hCorr
    global vCorr

    global magnitude
    
    # Create 2D lookup dictionaries to save the PTC and Correlation coefficients for plotting
    mean     = defaultdict(dict)
    PTC      = defaultdict(dict)
    groupPTC = defaultdict(dict)
    hCorr    = defaultdict(dict)
    vCorr    = defaultdict(dict)
    
    # Setup global statistics and filenames    
    statFlags = (afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV | afwMath.MAX | 
    afwMath.MIN | afwMath.ERRORS)
    print "The statistics flags are set to %s."%bin(statFlags)
    print "Errors will be calculated.\n"

    outDir       = 'output/lsst_flats_e_'
    suffix       = '_f2_R22_S11_E000.fits.gz'

    # Process Files
    magnitude = ['18', '15', '14', '13', '12', '10']
    extraId   = ['0', '1', '2', '3', '4']

    for (j, i) in itertools.product(extraId, magnitude):

        magnitude1  = i+'0'
        magnitude2  = i+'1'
        fileName1 = outDir+magnitude1+j+suffix
        fileName2 = outDir+magnitude2+j+suffix

        # Get images
        maskedImage1 = afwImg.ExposureF(fileName1).getMaskedImage()
        maskedImage2 = afwImg.ExposureF(fileName2).getMaskedImage()
        maskedImage3 = maskedImage1.clone()
        maskedImage3 -= maskedImage2

        # Process images
        if printLevel >= 1: print "Processing file ", fileName1
        (mean1, std1, groupMean1, groupStd1, hCorr1, vCorr1) = processImage(maskedImage1)
        if printLevel >= 1: print "Processing file ", fileName2
        (mean2, std2, groupMean2, groupStd2, hCorr2, vCorr2) = processImage(maskedImage2)
        if printLevel >= 1: print "Processing Difference"
        (mean3, std3, groupMean3, groupStd3, hCorr3, vCorr3) = processImage(maskedImage3)

        #Calculate PTC entry (Mean/Variance)
        PTC1      = mean1/std1**2
        PTC3      = (mean1+mean2)/std3**2
        groupPTC1 = groupMean1/groupStd1**2
        groupPTC3 = (groupMean1+groupMean2)/groupStd3**2

        # Print results
        if printLevel >= 1:
            print "\n---Results for magnitude", i, "config", j,":\n"

            print "Image1:\t %9.2f %9.2f %7.2f   "% (mean1, std1, PTC1)
            print "Image3:\t %9.2f %9.2f %7.2f \n"% (mean3, std3, PTC3)

            print "Group1:\t %9.2f %9.2f %7.2f   "% (groupMean1, groupStd1, groupPTC1)
            print "Group3:\t %9.2f %9.2f %7.2f \n"% (groupMean3, groupStd3, groupPTC3)

            print "Correlation1:\t %9.3f %9.3f"% (hCorr1, vCorr1)
            print "Correlation3:\t %9.3f %9.3f"% (hCorr3, vCorr3)
            print

        # Print Summary Line for this set of files
        print "%s %s %8.2f %7.2f %7.2f %9.3f %9.3f %7.2f %9.3f %9.3f"% (i, j, mean1, std1,
                                                                        PTC1, hCorr1, vCorr1,
                                                                        PTC3, hCorr3, vCorr3)

        # For Plotting
        mean[j][i]     = mean1
        PTC[j][i]      = PTC3
        groupPTC[j][i] = groupPTC3
        hCorr[j][i]    = hCorr3
        vCorr[j][i]    = vCorr3

    # Print the result to a file for use in plotting.  Use the SDSS shape output.
    outputFile =  open('flatData.py','w')

    for configuration in extraId:

        print >> outputFile, "numElectrons%s = [%s]" % \
        (configuration, ", ".join([str(mean[configuration][index]) for index in magnitude]))

        print >> outputFile, "PTC%s = [%s]" % \
        (configuration, ", ".join([str(PTC[configuration][index]) for index in magnitude]))

        print >> outputFile, "groupPTC%s = [%s]" % \
        (configuration, ", ".join([str(groupPTC[configuration][index]) for index in magnitude]))                
        print >> outputFile, "hCorr%s = [%s]" % \
        (configuration, ", ".join([str(hCorr[configuration][index]) for index in magnitude]))

        print >> outputFile, "vCorr%s = [%s]\n" % \
        (configuration, ", ".join([str(vCorr[configuration][index]) for index in magnitude]))

    outputFile.close()
                
if __name__ == "__main__":
    main()
