import matplotlib.pyplot as plt

#setup
printFigure = 1
saveFigure = 1
saveLink = "testErrorBar.png"


# example data
x = [20, 25, 30, 35, 40, 45, 50]
y = [1, 2, 2, 3, 5, 10, 7]
xerr = [0.6, 0.4, 0.2, 0.3, 0.2, 0.4, 0.4]
yerr = [0.5, 0.1, 0.4, 0.2, 0.2, 0.4, 0.4]


fig, ax = plt.subplots()
ax.errorbar(x, y, xerr, yerr)

if saveFigure == 1:
    plt.savefig(saveLink)

if printFigure == 1:
    plt.show()


