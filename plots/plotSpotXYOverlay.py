import matplotlib.pyplot as plt
import pandas as pd

h5store = pd.HDFStore('spotData.h5')
spots   = h5store['spots']

spotSizePlot, axis = plt.subplots(1,1)
spotSizePlot.suptitle('Standard Deviation in X and Y directions')

config0 = spots.query('config==0')
config4 = spots.query('config==4')

# Label is broken right now in pandas
config0.plot('numElectrons', 'ixx', yerr='errxx', ax=axis, fmt='ko', label='Perfect X')
config0.plot('numElectrons', 'iyy', yerr='erryy', ax=axis, fmt='bo', label='Perfect Y')
config4.plot('numElectrons', 'ixx', yerr='errxx', ax=axis, fmt="go", label='x500 X')
config4.plot('numElectrons', 'iyy', yerr='erryy', ax=axis, fmt="ro", label='x500 Y')

# Some bug(?) in Pandas means you have to this after plotting
axis.set_ylabel('IXX or IYY')
axis.set_xlabel('Number of Electrons')
axis.set_ylim(1.5, 1.9)
axis.set_xlim(0, 102000)
axis.legend(loc='best', prop={'size':12}, numpoints=1)
axis.grid('off', axis='both')

spotSizePlot.show()
h5store.close()
