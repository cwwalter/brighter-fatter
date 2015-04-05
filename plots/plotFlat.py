import matplotlib.pyplot as plt
import pandas as pd

def plotFrameVariable(dataFrame, title, variable, yLabel, yLim):

    config0 = dataFrame.query('config==0')
    config1 = dataFrame.query('config==1')
    config2 = dataFrame.query('config==2')
    config3 = dataFrame.query('config==3')
    config4 = dataFrame.query('config==4')

    flatPlot, axis = plt.subplots(1,1)
    flatPlot.suptitle(title)

    config0.plot('numElectrons', variable, style='ro', ax=axis, label='Perfect')
    config1.plot('numElectrons', variable, style='go', ax=axis, label='x1')
    config2.plot('numElectrons', variable, style='bo', ax=axis, label='x10')
    #config3.plot('numElectrons', variable, style='co', ax=axis, label='x100')
    #config3.plot('numElectrons', variable, style='yo', ax=axis, label='x500')

    axis.set_xlabel('Number of Electrons')
    axis.set_ylabel(yLabel)
    axis.set_xlim(0.0, 230000)
    axis.set_ylim(yLim)
    axis.grid('off', axis='both')
    axis.legend(loc='best', prop={'size':12}, numpoints=1)

    flatPlot.show()

# Main Program
def main():

    h5store = pd.HDFStore('flatData.h5')
    flats   = h5store['flats']
    
    plotFrameVariable(flats, 'PTC Curve for simulated flats',
                'PTC', 'Mean / Variance', (0.0, 2.8) )
    
    plotFrameVariable(flats, 'PTC Curve for simulated flats grouped into 4x4 pixels',
                'groupPTC', 'Mean / Variance', (0.0, 2.8) )
    
    plotFrameVariable(flats, 'Horizontal Autocorrelation Coefficients',
                'hCorr', 'Autocorrelation Coefficient', (-0.1, 0.15) )
    
    plotFrameVariable(flats, 'Vertical Autocorrelation Coefficients',
                'vCorr', 'Autocorrelation Coefficient', (-0.1, 0.15) )

    h5store.close()
    
if __name__ == "__main__":
    main()
