margins(0.05,.15)

numElectrons = [71,1127,2832,7113,17870]

hC0 = [ 0.009, .011, .011, .014, .014]
hC2 = [ 0.015, .026, .054, .275, .790]

diff0  = [.008, .013, .007, .013, .010]
diff2  = [.013, .016, .003, -.001, .001]


plot(numElectrons, hC0,    "bo", label='Perfect')
plot(numElectrons, hC2,   "ko", label='Charge Share')
plot(numElectrons, diff0, "ro", label='Perfect Diff')
plot(numElectrons, diff2, "go", label='Charge Share Diff')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Horizontal Autocorrelation Coefficients')
legend(loc='best', prop={'size':12})

ylim(-.1,1.0)

show()
