import matplotlib.pyplot    as plt
execfile('spotData.py')

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

xSize.errorbar(numElectrons, stdX0, yerr=errX0, fmt='ro', label='perfect')
ySize.errorbar(numElectrons, stdY0, yerr=errY0, fmt='ro', label='perfect')

xSize.errorbar(numElectrons, stdX1, yerr=errX1, fmt='go', label='x1')
ySize.errorbar(numElectrons, stdY1, yerr=errY1, fmt='go', label='x1')

xSize.errorbar(numElectrons, stdX2, yerr=errX2, fmt='bo', label='x10')
ySize.errorbar(numElectrons, stdY2, yerr=errY2, fmt='bo', label='x10')

xSize.errorbar(numElectrons, stdX3, yerr=errX3, fmt='co', label='x100')
ySize.errorbar(numElectrons, stdY3, yerr=errY3, fmt='co', label='x100')

xSize.errorbar(numElectrons, stdX4, yerr=errX4, fmt='yo', label='x500')
ySize.errorbar(numElectrons, stdY4, yerr=errY4, fmt='yo', label='x500')

ySize.legend(loc='best', prop={'size':5})
spotSizePlot.show()

