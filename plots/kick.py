close()

axis([-0.9,0.5,-0.9,0.5])

arrow(.15, .05, 0.0628343, -0.0628414)
arrow(.15, .10, 0.164363, -9.28627e-060)
arrow(.15, .15, 0.0628482, 0.0628482)
arrow(.10, .05, -9.28016e-06, -0.164275)
arrow(.10, .10, -0.810592, -0.810597)    
arrow(.10, .15, -9.28621e-06, 0.164363)
arrow(.05, .05, -0.0628276, -0.0628276)
arrow(.05, .10, -0.164275, -9.2802e-06)
arrow(.05, .15, -0.0628414, 0.0628343)

gca().add_patch(Rectangle((0, 0), .2, .2, facecolor="none"))

xlabel('X Pos')
ylabel('Y Pos')
title('Kick from PhoSim at 50000 electrons')
#legend(loc='best', prop={'size':12})

show()
