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
fields = ['centroid.naive', 
          #'centroid.naive.err', 'centroid.naive.flags',
          'centroid.gaussian', 
          #'centroid.gaussian.err',
          'centroid.sdss', 
          'centroid.sdss.flags',
          'shape.sdss', 
          'shape.sdss.centroid', 
          # 'shape.sdss.centroid.err',
          'shape.sdss.flags',
          'flux.gaussian']
algoKeys   = [schema.find(f).key for f in fields]

outDir       = 'output/lsst_e_'
suffix       = '_f2_R22_S11_E000.fits.gz'

numElectrons  = ['1000', '2000', '3000', '4000', '5000', 
                 '10000', '15000', '20000', '25000', '30000']
extraId = '2'

#numElectrons  = ['10000']

for i in numElectrons:

    print "using", i

    exposure    = afwImg.ExposureF(outDir+i+extraId+suffix)
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

    # Now loop through the keys we want
    #    for f,k in zip(fields, algoKeys):
    #        print '    ', f, source.get(k)
    
    for source in sources:
        print "Source found at ", source.getCentroid()
        x,y = source.getCentroid()

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
        print "Electrons = %2d STDX = %4.3f STDY = %4.3f \n" % \
            (int(i), stdx, stdy)


