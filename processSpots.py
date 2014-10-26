# DM processing of Phosim output
# You should 'setup pipe_test' to use it.
# This program
# C. Walter 01/2014

from collections import defaultdict

import math
import itertools
import numpy                as np

import lsst.pex.logging     as pexLog
import lsst.afw.math        as afwMath
import lsst.afw.table       as afwTable
import lsst.afw.image       as afwImg
import lsst.afw.detection   as afwDetect

import lsst.meas.algorithms as measAlg

statFlags = (afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV | afwMath.MAX | 
             afwMath.MIN | afwMath.ERRORS)
print "The statistics flags are set to %s."%bin(statFlags)
print "Errors will be calculated.\n"

# Configure the detection and measurement algorithms
schema                = afwTable.SourceTable.makeMinimalSchema()
detectSourcesConfig   = measAlg.SourceDetectionConfig(thresholdType='value')
measureSourcesConfig  = measAlg.SourceMeasurementConfig()

# Setup the detection and measurement tasks
detect  = measAlg.SourceDetectionTask(config=detectSourcesConfig,  
                                      schema=schema)
measure = measAlg.SourceMeasurementTask(config=measureSourcesConfig, 
                                        schema=schema)

#  Choose algorithms to look at the output of. 
fields = [#'centroid.naive', 
          #'centroid.naive.err', 'centroid.naive.flags',
          #'centroid.gaussian', 
          #'centroid.gaussian.err',
          'centroid.sdss', 
          'centroid.sdss.flags',
          'shape.sdss',
          'shape.sdss.err', 
          'shape.sdss.centroid', 
          # 'shape.sdss.centroid.err',
          'shape.sdss.flags',
          'flux.gaussian']

algoKeys   = [schema.find(f).key for f in fields]

outDir       = 'output/lsst_e_'
suffix       = '_f2_R22_S11_E000.fits.gz'

numElectrons  = ['1000', '2000', '3000', '4000', '5000', 
                 '10000', '15000', '20000', '25000', '30000',
                 '50000', '75000', '100000']

extraId = ['0','1','2','3','4']
idColor = {'0':'ro', '1':'go', '2':'bo', '3':'co', '4':'yo'}
#extraId = ['0', '1']

# Create 2D lookup dictionaries to save the calculated spot widths
stdX = defaultdict(dict)
stdY = defaultdict(dict)

# Set the verbosity level
printLevel = 0
pexLog.Log.getDefaultLog().setThreshold(pexLog.Log.WARN)

# Loop over the set of files in electron intensity and BF effect strength
for (j, i) in itertools.product(extraId, numElectrons):

    if printLevel >= 1:
        print "using", j, i

    exposure    = afwImg.ExposureF(outDir+i+j+suffix)
    maskedImage = exposure.getMaskedImage()
    
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
        print "Mean   = %9.8f +- %3.1f" %imageStatistics.getResult(afwMath.MEAN)
        print "StdDev = %9.2f"          %imageStatistics.getResult(afwMath.STDEV)[0]
    
    # Detect the sources,then put them into a catalog 
    # (the table is where the catalog atually stores stuff)
    table   = afwTable.SourceTable.make(schema)
    catalog = detect.makeSourceCatalog(table, exposure, sigma=5)

    # Get the sources out of the catalog
    sources = catalog.sources
    
    # Apply the measurement routines to the exposure using the sources as input
    measure.run(exposure, sources)

    # For now there should only be one source (otherwise this won't work right)
    for source in sources:
        if printLevel >= 1:
            print "Source found at ", source.getCentroid()

        x,y = source.getCentroid()

        # Now loop through the keys we want
        for f,k in zip(fields, algoKeys):
            #print '    ', f, source.get(k)
            if f=='shape.sdss':
                ixx = math.sqrt(source.get(k).getIxx())
                iyy = math.sqrt(source.get(k).getIyy())
                ixy = source.get(k).getIxy()
                stdX[j][i] = ixx
                stdY[j][i] = iyy
            #if f=='shape.sdss.err':            
                #print "ERROR!"

        # Calculate the sizes myself by oversampling.  This is
        # necessary for single pixel size spots where the default algorithms fail.

        # Had to add this for dev branch
        x = round (x)
        y = round (y)
                
        yaxis      = np.linspace(y-10,y+10,200)
        yvalues    = np.repeat(image.getArray()[y-10:y+10,x],10)
        myaverage  = np.average(yaxis, weights=yvalues)
        variancey  = np.average((yaxis-myaverage)**2, weights=yvalues)
        stdy       = math.sqrt(variancey)

        xaxis      = np.linspace(x-10,x+10,200)
        xvalues    = np.repeat(image.getArray()[y,x-10:x+10],10)
        myaverage =  np.average(xaxis, weights=xvalues)
        variancex  = np.average((xaxis-myaverage)**2, weights=xvalues)
        stdx       = math.sqrt(variancex)
        
        print "ID:%1s Electrons= %6s STDX= %4.2f STDY= %4.2f IXX= %4.2f IYY= %4.2f IXY= %4.2f" % \
            (j, i, stdx, stdy, ixx, iyy, ixy)

# Print the result to a file for use in plotting.  Use the SDSS shape output.
outputFile =  open('plotSpot.py','w')

print >> outputFile, """
import matplotlib.pyplot    as plt

spotSizePlot = plt.figure()
spotSizePlot.suptitle('Standard Deviation in X and Y directions')

xSize = spotSizePlot.add_subplot(211)
xSize.set_ylabel('Sigma X')
xSize.margins(0.05,.15)

ySize = spotSizePlot.add_subplot(212)
ySize.set_ylabel('Sigma Y')
ySize.set_xlabel('Number of Electrons')
ySize.margins(0.05,.15)
"""
print >> outputFile, "numElectrons =", numElectrons, "\n"

for configuration in extraId:

    print >> outputFile, "stdX%s = [%s]" % \
    (configuration, ", ".join([str(stdX[configuration][electron]) for electron in numElectrons]))
    
    print >> outputFile, "stdY%s = [%s]\n" % \
    (configuration, ", ".join([str(stdY[configuration][electron]) for electron in numElectrons]))

    print >> outputFile, 'xSize.plot(numElectrons, stdX'+configuration+',"'+idColor[configuration]+'")'
    print >> outputFile, 'ySize.plot(numElectrons, stdY'+configuration+',"'+idColor[configuration]+'")\n'
        
print >> outputFile, """
spotSizePlot.show()
"""
