import matplotlib.pyplot as plt

#https://matplotlib.org/users/pyplot_tutorial.html

#graph function
def drawGraph( xList, yList, xLabel, yLabel):
    plt.plot(xList, yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()
    return

#test program
xList = [1,2,3,4]
yList = [1,10,50,100]
xLabel = "x Achse"
yLabel = "y Achse"
drawGraph( xList, yList, xLabel, yLabel)
