import numpy as np
import matplotlib.pyplot as plt


def drawGraph( xLabel, yLabel, printFigure, saveFigure, saveLink, *args ):

    plt.plot(*args)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

    return
