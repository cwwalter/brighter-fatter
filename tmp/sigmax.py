

numElectrons = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000]
stdX         = [0.289, 0.290, 0.292, 0.292, 0.292, 0.295, 0.298, 0.301, 0.302, 0.306]
stdY         = [0.289, 0.289, 0.289, 0.289, 0.289, 0.289, 0.289, 0.289, 0.289, 0.289]

plot(numElectrons, stdX, "ro")
xlabel('Number of Electrons')
ylabel('Sigma X')
title('Standard Deviation in X direction')
