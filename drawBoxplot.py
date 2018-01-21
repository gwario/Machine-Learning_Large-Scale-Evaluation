import matplotlib.pyplot as plt
import numpy as np
import pprint

def drawBoxplotbar(xLabel, yLabel, printFigure, saveFigure, saveLink, data, labels):

    plt.boxplot(data, labels=labels)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    if saveFigure == 1:
        print("Boxplot saved as {}".format(saveLink))
        plt.savefig(saveLink)
        
    if printFigure == 1:
        plt.show()

    plt.close()
    return
    
    

