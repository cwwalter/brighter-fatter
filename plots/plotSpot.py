import matplotlib.pyplot  as plt
from matplotlib.pyplot import *
from astropy.table import Table

data = Table.read('spotData.fits')

spotSizePlot = plt.figure()
spotSizePlot.suptitle('Standard Deviation in X and Y directions')

xSize = spotSizePlot.add_subplot(211)
xSize.set_ylabel('Sigma X')
#xSize.margins(0.05,.15)
xSize.set_xlim(0,102000)
xSize.set_ylim(1.5,1.9)

ySize = spotSizePlot.add_subplot(212)
ySize.set_ylabel('Sigma Y')
ySize.set_xlabel('Number of Electrons')
#ySize.margins(0.05,.15)
ySize.set_xlim(0,102000)
ySize.set_ylim(1.5,1.9)

xSize.errorbar(data['numElectrons'][0], data['stdX'][0], yerr=data['errX'][0], fmt='ro', label='perfect')
ySize.errorbar(data['numElectrons'][0], data['stdY'][0], yerr=data['errY'][0], fmt='ro', label='perfect')

xSize.errorbar(data['numElectrons'][1], data['stdX'][1], yerr=data['errX'][1], fmt='go', label='x1')
ySize.errorbar(data['numElectrons'][1], data['stdY'][1], yerr=data['errY'][1], fmt='go', label='x1')

xSize.errorbar(data['numElectrons'][2], data['stdX'][2], yerr=data['errX'][2], fmt='bo', label='x10')
ySize.errorbar(data['numElectrons'][2], data['stdY'][2], yerr=data['errY'][2], fmt='bo', label='x10')

xSize.errorbar(data['numElectrons'][3], data['stdX'][3], yerr=data['errX'][3], fmt='co', label='x100')
ySize.errorbar(data['numElectrons'][3], data['stdY'][3], yerr=data['errY'][3], fmt='co', label='x100')

xSize.errorbar(data['numElectrons'][3], data['stdX'][4], yerr=data['errX'][4], fmt='yo', label='x500')
ySize.errorbar(data['numElectrons'][3], data['stdY'][4], yerr=data['errY'][4], fmt='yo', label='x500')

ySize.legend(loc=(.82,.5), prop={'size':9}, numpoints=1)
spotSizePlot.show()

