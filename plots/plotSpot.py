import matplotlib.pyplot as plt
import pandas as pd

h5store = pd.HDFStore('spotData.h5')
spots = h5store['spots']


def setPercentageAxisLimits(axis, setPoint):
    """
    Return the limits of the secondary y-axis in percentage RMS change.
    The input is the left-hand axis and the no-BF effect RMS.
    """
    y1, y2 = axis.get_ylim()
    lowerLimit = (y1 - setPoint)/setPoint*100.0
    upperLimit = (y2 - setPoint)/setPoint*100.0

    return (lowerLimit, upperLimit)

spotSizePlot, (xPlot, yPlot) = plt.subplots(2, 1)
spotSizePlot.suptitle('Standard Deviation in X and Y directions', fontsize=15)

config0 = spots.query('config==0')
config1 = spots.query('config==1')
config2 = spots.query('config==2')
config3 = spots.query('config==3')
config4 = spots.query('config==4')

config0.plot('maxValue', 'ixx', yerr='errxx', fmt='ro', ax=xPlot, label='Perfect')
config1.plot('maxValue', 'ixx', yerr='errxx', fmt='go', ax=xPlot, label='Nominal BF')
# config2.plot('maxValue', 'ixx', yerr='errxx', fmt='bo', ax=xPlot, label='x10')
# config3.plot('maxValue', 'ixx', yerr='errxx', fmt='co', ax=xPlot, label='x100')
# config4.plot('maxValue', 'ixx', yerr='errxx', fmt='yo', ax=xPlot, label='x500')

config0.plot('maxValue', 'iyy', yerr='erryy', fmt='ro', ax=yPlot, label='Perfect')
config1.plot('maxValue', 'iyy', yerr='erryy', fmt='go', ax=yPlot, label='Nominal BF')
# config2.plot('maxValue', 'iyy', yerr='erryy', fmt='bo', ax=yPlot, label='x10')
# config3.plot('maxValue', 'iyy', yerr='erryy', fmt='co', ax=yPlot, label='x100')
# config4.plot('maxValue', 'iyy', yerr='erryy', fmt='yo', ax=yPlot, label='x500')

# xPlot.grid('on', axis='both')
# yPlot.grid('on', axis='both')

xPlot.set_xlabel('')
xPlot.set_ylabel('Sigma X')
xPlot.set_xlim(0, 175000)
xPlot.set_ylim(1.6, 1.7)
xPlot.legend(loc=(.8, .77), prop={'size': 9}, numpoints=1)

yPlot.set_xlabel('Maximum Number of Electrons in a Pixel')
yPlot.set_ylabel('Sigma Y')
yPlot.set_xlim(0, 175000)
yPlot.set_ylim(1.6, 1.7)
yPlot.legend().remove()

# Make 2nd y-axes on right-hand side of plot
xPlotPercentage = xPlot.twinx()
xPlotPercentage.set_ylim(setPercentageAxisLimits(xPlot, 1.624))
xPlotPercentage.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
xPlotPercentage.set_ylabel('% BF')

yPlotPercentage = yPlot.twinx()
yPlotPercentage.set_ylim(setPercentageAxisLimits(yPlot, 1.624))
yPlotPercentage.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
yPlotPercentage.set_ylabel('% BF')

xPlotPercentage.plot([0.0, 200000.0], [0.0, 0.0], color='grey', linestyle='--')
yPlotPercentage.plot([0.0, 200000.0], [0.0, 0.0], color='grey', linestyle='--')

spotSizePlot.show()
h5store.close()
