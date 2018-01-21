import matplotlib.pyplot as plt
import numpy as np

def drawBoxplotbar(xLabel, yLabel, printFigure, saveFigure, saveLink, *args):

    if len(args) >= 4:
        
        d = np.concatenate((args[0], args[1], args[2], args[3]), 0)
        data = [d]

        if len(args) >= 8:

            for i in range(1, int(len(args)/4)):
                d = np.concatenate((args[i], args[i+1], args[i+2], args[i+3]), 0)
                data.append(d)

    plt.boxplot(data)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    if saveFigure == 1:
        plt.savefig(saveLink)
        
    if printFigure == 1:
        plt.show()


    return
    
    

