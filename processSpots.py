# DM processing of Phosim spot output
# You should 'setup pipe_test' to use it.
#
# C. Walter: last major update to use DM v10.1 - 06/2015

import math
import numpy as np
import pandas as pd
import itertools

import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg

import lsst.meas.algorithms as measAlg
import lsst.meas.base as measBase
import lsst.pex.logging as pexLog

# Set the program verbosity level
printLevel = 0
pexLog.Log.getDefaultLog().setThreshold(pexLog.Log.WARN)

# Turn off warning due to reading in an image into an Exposure
# This should be temporary until DM fixes this particular issue.
# https://jira.lsstcorp.org/browse/DM-3191
pexLog.Log.getDefaultLog().setThresholdFor("afw.image.MaskedImage",
                                           pexLog.Log.FATAL)

# Make a Panda hd5f data store and frame.
h5store = pd.HDFStore('spotData.h5', mode='w')
spots = pd.DataFrame(columns=('config', 'numElectrons', 'maxValue',
                     'ixx', 'errxx', 'iyy', 'erryy', 'stdx', 'stdy'))

# File info
outDir = 'output/lsst_e_'
suffix = '_f2_R22_S11_E000.fits.gz'

electronLevel = [1000, 2000, 3000, 4000, 5000,
                 10000, 15000, 20000, 25000, 30000,
                 50000, 75000, 100000, 200000, 500000, 750000,
                 1000000, 1250000, 1500000, 1750000, 2000000]

# extraId = [0, 1, 2, 3, 4]
extraId = [0, 1]

# Set the statistics flags
statFlags = (afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV | afwMath.MAX |
             afwMath.MIN | afwMath.ERRORS)
print "The statistics flags are set to %s." % bin(statFlags)
print "Errors will be calculated.\n"

# Configure the detection and measurement algorithms
schema = afwTable.SourceTable.makeMinimalSchema()
detectSourcesConfig = measAlg.SourceDetectionConfig(thresholdType='value')
measureSourcesConfig = measBase.SingleFrameMeasurementConfig()

# Choose algorithms to use
measureSourcesConfig.plugins.names = ["base_GaussianCentroid",
                                      "base_SdssShape"]
measureSourcesConfig.slots.centroid = "base_GaussianCentroid"

# Set flux aliases to None. This is necessary since by defualt these point at
# some alogorithims that are not in my list above.
measureSourcesConfig.slots.psfFlux = None
measureSourcesConfig.slots.apFlux = None
measureSourcesConfig.slots.modelFlux = None
measureSourcesConfig.slots.instFlux = None
measureSourcesConfig.slots.calibFlux = None
measureSourcesConfig.validate()

# Setup the detection and measurement tasks
detect = measAlg.SourceDetectionTask(config=detectSourcesConfig,
                                     schema=schema)
measure = measBase.SingleFrameMeasurementTask(config=measureSourcesConfig,
                                              schema=schema)
#  Choose algorithms to look at the output of.
fields = ['base_GaussianCentroid', 'base_SdssShape']

# Find the keys for those algorithms.
algoKeys = []
for f in fields:
    if f == 'base_GaussianCentroid':
        k = afwTable.Point2DKey(schema[f])
    elif f == 'base_SdssShape':
        k = measBase.SdssShapeResultKey(schema[f])
    else:
        k = schema.find(f).key
    algoKeys.append(k)

if printLevel >= 2:
    for i, key in enumerate(zip(fields, algoKeys)):
        print i, key

# Loop over the set of files in electron intensity and BF effect strength
for (j, i) in itertools.product(extraId, electronLevel):

    if printLevel >= 1:
        print "using", j, i

    # Make a afw Exposure
    fileName = "%s%02d%d%s" % (outDir, i, j, suffix)
    exposure = afwImg.ExposureF(fileName)

    # Add a Gaussian PSF to the exposure.  This is needed by the SDSS Centroid
    # Algorithm.  For the kernel make it 3*sigma on either side of the
    # central pixel.
    sigma = 1.0
    size = 2*int(3*sigma) + 1
    gaussianPSF = measAlg.SingleGaussianPsf(size, size, sigma)
    exposure.setPsf(gaussianPSF)

    # Make a maskedImage
    maskedImage = exposure.getMaskedImage()

    # These three are held in the maskedImage
    image = maskedImage.getImage()
    mask = maskedImage.getMask()
    variance = maskedImage.getVariance()

    # We need to manually make the variance plane since it is not in the
    # phosim image.
    variance.getArray()[:, :] = np.abs(image.getArray())

    imageStatistics = afwMath.makeStatistics(maskedImage, statFlags)
    numBins = imageStatistics.getResult(afwMath.NPOINT)[0]
    mean = imageStatistics.getResult(afwMath.MEAN)[0]
    maxValue = imageStatistics.getResult(afwMath.MAX)[0]

    if printLevel >= 2:
        print "The image has dimensions %i x %i pixels" \
            % (maskedImage.getWidth(), maskedImage.getHeight())
        print "Number of analyzed bins in image is %i" % numBins
        print "Max  = %9d" % imageStatistics.getResult(afwMath.MAX)[0]
        print "Min  = %9d" % imageStatistics.getResult(afwMath.MIN)[0]
        print "Mean = %9.8f +- %3.1f" % imageStatistics.getResult(afwMath.MEAN)
        print "StdDev = %9.2f" % imageStatistics.getResult(afwMath.STDEV)[0]

    # Detect the sources,then put them into a catalog
    # (the table is where the catalog atually stores stuff)
    table = afwTable.SourceTable.make(schema)
    catalog = detect.makeSourceCatalog(table, exposure, sigma=5)

    # Get the sources out of the catalog
    sources = catalog.sources

    # Apply the measurement routines to the exposure using the sources as input
    measure.run(exposure, sources)

    # For now there should only be one source (otherwise this won't work right)
    for source in sources:
        if printLevel >= 1:
            print "Source found at ", source.getCentroid()

    x, y = source.getCentroid()

    # Now loop through the keys we want
    for f, k in zip(fields, algoKeys):
        # print '    ', f, source.get(k)
        if f == 'base_SdssShape':
            result = source.get(k)

            # Currently bug in DM sets this flag every time
            # errorFlag = result.getFlag(measBase.SdssShapeAlgorithm.FAILURE)
            # if errorFlag is True:
            #     print "Error on File!"

            if printLevel >= 2:
                print dir(result)
                print result.x, result.y
                print result.getCentroid()
                print result.xx, result.yy, result.xy
                print result.xxSigma, result.yySigma
                print result.getShape()
                print result.getShapeErr()
                print result.getQuadrupole()

            ixx = math.sqrt(result.xx)
            iyy = math.sqrt(result.yy)
            ixy = result.xy  # This could be be negative

            errxx = result.xxSigma
            erryy = result.yySigma
            errxy = result.xySigma

    # Calculate the sizes myself by oversampling.  This is
    # necessary for single pixel size spots where the default algorithms fail.

    yaxis = np.linspace(y - 10, y + 10, 200)
    yvalues = np.repeat(image.getArray()[y - 10:y + 10, x], 10)
    myaverage = np.average(yaxis, weights=yvalues)
    variancey = np.average((yaxis - myaverage)**2, weights=yvalues)
    stdy = math.sqrt(variancey)

    xaxis = np.linspace(x - 10, x + 10, 200)
    xvalues = np.repeat(image.getArray()[y, x - 10:x + 10], 10)
    myaverage = np.average(xaxis, weights=xvalues)
    variancex = np.average((xaxis - myaverage)**2, weights=xvalues)
    stdx = math.sqrt(variancex)

    print ("ID:%1d Electrons= %7s STDX= %4.2f STDY= %4.2f IXX= %4.2f "
           "IYY= %4.2f IXY= %4.2f") % (j, i, stdx, stdy, ixx, iyy, ixy)

    # Fill the Pandas data frame
    spots.loc[len(spots)] = (j, i, maxValue, ixx, errxx, iyy, erryy,
                             stdx, stdy)

# Write out the data store
h5store['spots'] = spots
h5store.close()
