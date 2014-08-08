margins(0.05,.15)

numElectrons = [71,1127,2832,7113,17870]

vC0 = [ 0.008, .010, .014, .013, .010]
vC2 = [ 0.011, .024, .050, .267, .779]

diff0  = [.008, .010, .013, .009, .007]
diff2  = [.011, .007, .009, .007, .003]

plot(numElectrons, vC0,    "bo", label='Perfect')
plot(numElectrons, vC2,   "ko", label='Charge Share')
plot(numElectrons, diff0, "ro", label='Perfect Diff')
plot(numElectrons, diff2, "go", label='Charge Share Diff')

xlabel('Number of Electrons')
ylabel('Autocorrelation Coefficient')
title('Vertical Autocorrelation Coefficients')
legend(loc='best', prop={'size':12})

ylim(-.1,1.0)

show()
