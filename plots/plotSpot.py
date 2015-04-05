import matplotlib.pyplot  as plt
import pandas as pd

h5store = pd.HDFStore('spotData.h5')
spots   = h5store['spots']

spotSizePlot, (xPlot, yPlot) = plt.subplots(2,1)
spotSizePlot.suptitle('Standard Deviation in X and Y directions')

config0 = spots.query('config==0')
config1 = spots.query('config==1')
config2 = spots.query('config==2')
config3 = spots.query('config==3')
config4 = spots.query('config==4')

config0.plot('numElectrons', 'ixx', yerr='errxx', fmt='ro', ax=xPlot, label='perfect')
config1.plot('numElectrons', 'ixx', yerr='errxx', fmt='go', ax=xPlot, label='x1')
config2.plot('numElectrons', 'ixx', yerr='errxx', fmt='bo', ax=xPlot, label='x10')
config3.plot('numElectrons', 'ixx', yerr='errxx', fmt='co', ax=xPlot, label='x100')
config4.plot('numElectrons', 'ixx', yerr='errxx', fmt='yo', ax=xPlot, label='x500')

config0.plot('numElectrons', 'iyy', yerr='erryy', fmt='ro', ax=yPlot, label='perfect')
config1.plot('numElectrons', 'iyy', yerr='erryy', fmt='go', ax=yPlot, label='x1')
config2.plot('numElectrons', 'iyy', yerr='erryy', fmt='bo', ax=yPlot, label='x10')
config3.plot('numElectrons', 'iyy', yerr='erryy', fmt='co', ax=yPlot, label='x100')
config4.plot('numElectrons', 'iyy', yerr='erryy', fmt='yo', ax=yPlot, label='x500')

xPlot.set_ylabel('Sigma X')
xPlot.set_xlim(0,102000)
xPlot.set_ylim(1.5,1.9)
xPlot.grid('off', axis='both')
xPlot.legend().remove()

yPlot.set_xlabel('Number of Electrons')
yPlot.set_ylabel('Sigma Y')
yPlot.set_xlim(0,102000)
yPlot.set_ylim(1.5,1.9)
yPlot.grid('off', axis='both')
yPlot.legend(loc=(.82,.5), prop={'size':9}, numpoints=1)

spotSizePlot.show()
h5store.close()
