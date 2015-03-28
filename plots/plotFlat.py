from matplotlib.pyplot import *
from astropy.table import Table

data = Table.read('flatData.fits')

# Make PTC plot

margins(0.05,.15)

plot(data['numElectrons'][0], data['PTC'][0],  "ro", label='Perfect')
plot(data['numElectrons'][1], data['PTC'][1],  "go", label='x1')
plot(data['numElectrons'][2], data['PTC'][2],  "bo", label='x10')
#plot(data['numElectrons'][0], data['PTC'][3],  "co", label='x100')
#plot(data['numElectrons'][0], data['PTC'][4],  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Mean / Variance')
title('PTC Curve for simulated flats')
legend(loc='best', prop={'size':12}, numpoints=1)

ylim(0.0,2.8)

show()

# Make group 4x4 PTC plot
figure()

margins(0.05,.15)

plot(data['numElectrons'][0], data['groupPTC'][0],  "ro", label='Perfect')
plot(data['numElectrons'][1], data['groupPTC'][1],  "go", label='x1')
plot(data['numElectrons'][2], data['groupPTC'][2],  "bo", label='x10')
#plot(data['numElectrons'][3], data['PTC'][3],  "co", label='x100')
#plot(data['numElectrons'][4], data['PTC'][4],  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Mean / Variance')
title('PTC Curve for simulated flats grouped into 4x4 pixels')
legend(loc='best', prop={'size':12}, numpoints=1)

ylim(0.0,2.8)

show()

# Make horizontal correlation coefficient plot

figure()

margins(0.05,.15)

plot(data['numElectrons'][0], data['hCorr'][0],  "ro", label='Perfect')
plot(data['numElectrons'][1], data['hCorr'][1],  "go", label='x1')
plot(data['numElectrons'][2], data['hCorr'][2],  "bo", label='x10')
##plot(data['numElectrons'][3], data['hCorr'][3],  "co", label='x100')
##plot(data['numElectrons'][4], data['hCorr'][4],  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Horizontal Autocorrelation Coefficients')
legend(loc='best', prop={'size':12}, numpoints=1)

ylim(-.1,0.15)

# Make vertical correlation coefficient plot

figure()

plot(data['numElectrons'][0], data['vCorr'][0],  "ro", label='Perfect')
plot(data['numElectrons'][1], data['vCorr'][1],  "go", label='x1')
plot(data['numElectrons'][2], data['vCorr'][2],  "bo", label='x10')
##plot(data['numElectrons'][3], data['vCorr'][3],  "co", label='x100')
##plot(data['numElectrons'][4], data['vCorr'][4],  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Vertical Autocorrelation Coefficients')
legend(loc='best', prop={'size':12}, numpoints=1)

ylim(-.1,0.15)

show()
