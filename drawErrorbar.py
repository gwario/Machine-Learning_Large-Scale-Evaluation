import matplotlib.pyplot as plt

def drawErrorbar( xList, yList, xErrList, yErrList, xLabel, yLabel, printFigure, saveFigure, saveLink):

    fig, ax = plt.subplots()
    ax.errorbar(xList, yList, xErrList, yErrList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)


    if saveFigure == 1:
        plt.savefig(saveLink)

    if printFigure == 1:
        plt.show()
        
    plt.close()

    return












