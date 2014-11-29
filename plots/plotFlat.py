import matplotlib.pyplot as plt

execfile('flatData.py')

# Make PTC plot

margins(0.05,.15)

plot(numElectrons0, PTC0,  "ro", label='Perfect')
plot(numElectrons0, PTC1,  "go", label='x1')
plot(numElectrons0, PTC2,  "bo", label='x10')
#plot(numElectrons0, PTC3,  "co", label='x100')
#plot(numElectrons0, PTC4,  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Mean / Variance')
title('PTC Curve for simulated flats')
legend(loc='best', prop={'size':12})

ylim(0.0,2.8)

show()

# Make group 4x4 PTC plot
figure()

margins(0.05,.15)

plot(numElectrons0, groupPTC0,  "ro", label='Perfect')
plot(numElectrons0, groupPTC1,  "go", label='x1')
plot(numElectrons0, groupPTC2,  "bo", label='x10')
#plot(numElectrons0, PTC3,  "co", label='x100')
#plot(numElectrons0, PTC4,  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Mean / Variance')
title('PTC Curve for simulated flats grouped into 4x4 pixels')
legend(loc='best', prop={'size':12})

ylim(0.0,2.8)

show()

# Make horizontal correlation coefficient plot

figure()

margins(0.05,.15)

plot(numElectrons0, hCorr0,  "ro", label='Perfect')
plot(numElectrons0, hCorr1,  "go", label='x1')
plot(numElectrons0, hCorr2,  "bo", label='x10')
##plot(numElectrons0, hCorr3,  "co", label='x100')
##plot(numElectrons0, hCorr4,  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Horizontal Autocorrelation Coefficients')
legend(loc='best', prop={'size':12})

ylim(-.1,0.15)

# Make vertical correlation coefficient plot

figure()

plot(numElectrons0, vCorr0,  "ro", label='Perfect')
plot(numElectrons0, vCorr1,  "go", label='x1')
plot(numElectrons0, vCorr2,  "bo", label='x10')
##plot(numElectrons0, vCorr3,  "co", label='x100')
##plot(numElectrons0, vCorr4,  "yo", label='x500')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Vertical Autocorrelation Coefficients')
legend(loc='best', prop={'size':12})

ylim(-.1,0.15)

show()
