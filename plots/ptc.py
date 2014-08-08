margins(0.05,.15)

numElectrons = [71,1127,2832,7113,17870]

ptc0  = [ 1.01, 1.01, 1.01, 1.01, 1.02]
diff0 = [ 1.01, 1.01, 1.01, 1.01, 1.01]

ptc2  = [ 1.01, 1.02, 1.06, 1.40, 4.99]
diff2 = [ 1.01, 1.01, 1.01, 1.01, 1.00]


plot(numElectrons, ptc0,  "bo", label='Perfect')
plot(numElectrons, diff0, "ko", label='Perfect Diff')
plot(numElectrons, ptc2,  "ro", label='Charge Share')
plot(numElectrons, diff2, "go", label='Charge Share diff')
xlabel('Number of Electrons')
ylabel('Variance / Mean')
title('PTC Curve for simulated flats')
legend(loc='best', prop={'size':12})

ylim(-1,6)

show()
