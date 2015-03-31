import matplotlib.pyplot    as plt
from matplotlib.pyplot import *
from astropy.table import Table

data = Table.read('spotData.fits')


spotSizePlot = plt.figure()
spotSizePlot.suptitle('Standard Deviation in X and Y directions')

xSize = spotSizePlot.add_subplot(111)
xSize.set_ylabel('Sigma')
#xSize.margins(0.05,.15)
xSize.set_xlim(0,102000)
xSize.set_ylim(1.5,1.9)

xSize.errorbar(data['numElectrons'][0], data['stdX'][0], yerr=data['errX'][0], fmt='ko', label='perfect X')
xSize.errorbar(data['numElectrons'][0], data['stdY'][0], yerr=data['errY'][0], fmt='bo', label='perfect Y')

xSize.errorbar(data['numElectrons'][4], data['stdX'][4], yerr=data['errX'][4], fmt='go', label='x500 X')
xSize.errorbar(data['numElectrons'][4], data['stdY'][4], yerr=data['errY'][4], fmt='ro', label='x500 Y')

xSize.legend(loc='best', prop={'size':12},numpoints=1)
spotSizePlot.show()

