margins(0.05,.15)

numElectrons = [71,1127,2832,7113,17870]

ptc0  = [ 1.16, 1.17, 1.17, 1.20, 1.21]
ptc2  = [ 1.17, 1.32, 1.83, 6.43, 57.94]

plot(numElectrons, ptc0,  "bo", label='Perfect')
plot(numElectrons, ptc2,  "ro", label='Charge Share')
xlabel('Number of Electrons')
ylabel('Variance / Mean')
title('PTC Curve for simulated flats grouped into 4x4 blocks')
legend(loc='best', prop={'size':12})

ylim(-1,60)

show()
