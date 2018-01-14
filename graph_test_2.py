import numpy as np
import matplotlib.pyplot as plt

#https://matplotlib.org/users/pyplot_tutorial.html

#graph function
def drawGraph( xList, yList, xLabel, yLabel):
    fig = plt.figure()
    plt.plot(xList, yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    fig.show()
    return

#test program
xList = [1,2,3,4,9]
yList = [1,10,50,100,30]
xList2 = [1, 3, 9]
yList2 = [20, 120, 140]
xLabel = "x Achse"
yLabel = "y Achse"
drawGraph( xList, yList, xLabel, yLabel)
#new
fig1 = plt.figure()
t = np.arange(0., 9., 0.2)
plt.plot(xList, yList, 'r--', xList2, yList2, 'bs', t, t**3, 'g^')
fig1.show()
