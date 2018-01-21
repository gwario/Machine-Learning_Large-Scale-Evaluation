import drawErrorbar as eb
import drawBoxplot as bp
import drawGraph as g

import matplotlib.pyplot as plt
import numpy as np





# example text
xLabel = "x label"
yLabel = "y label"

# example setup
printFigure = 1 #0 = don't / 1 = do
saveFigure = 0  #0 = don't / 1 = do
saveLink = "testFigureFilepath.png"

# example - data please make sure that all lists have the same length
xList = [20, 25, 30, 35, 40, 45, 50]
yList = [1, 2, 2, 3, 5, 10, 7]
xErr = [0.6, 0.4, 0.2, 0.3, 0.2, 0.4, 0.4]
yErr = [0.5, 0.1, 0.4, 0.2, 0.2, 0.4, 0.4]

######## FUNCTION CALL ERRORBAR ######## - you can make the lists as long as you want them
eb.drawErrorbar(xList, yList, xErr, yErr, xLabel, yLabel, printFigure, saveFigure, saveLink)





# more example data - for each dataset you need spread, center, flier_high, and flier_low
spread1 = np.random.rand(10) * 100
center1 = np.ones(25) * 50
flier_high1 = np.random.rand(10) * 100 + 100
flier_low1 = np.random.rand(10) * -100

spread2 = np.random.rand(50) * 100
center2 = np.ones(25) * 40
flier_high2 = np.random.rand(10) * 100 + 100
flier_low2 = np.random.rand(10) * -100

spread3 = np.random.rand(10) * 100
center3 = np.ones(25) * 50
flier_high3 = np.random.rand(10) * 100 + 100
flier_low3 = np.random.rand(10) * -100

spread4 = np.random.rand(50) * 100
center4 = np.ones(25) * 40
flier_high4 = np.random.rand(10) * 100 + 100
flier_low4 = np.random.rand(10) * -100

######## FUNCTION CALL BOXPLOT ######## - you can insert as many datasets as you want, in this example we insert 4
bp.drawBoxplotbar(xLabel, yLabel, printFigure, saveFigure, saveLink, spread1, center1, flier_high1, flier_low1, spread2, center2, flier_high2, flier_low2, spread3, center3, flier_high3, flier_low3, spread4, center4, flier_high4, flier_low4)





# more example data
xList2 = [1,2,3,4,5,6,7,8,9,10,11,12]
yList2 = [1,10,50,100,1,10,50,100,1,10,50,100]

xList3 = [1,2,3,4,5,6,7,8,9,10,11,12]
yList3 = [200,180,170,160,200,180,170,160,200,180,170,160,]

######## FUNCTION CALL BOXPLOT ######## - you can insert as many datasets as you want, in this example we insert 4 - if you want, you can insert color/line options after each dataset
g.drawGraph(xLabel, yLabel, printFigure, saveFigure, saveLink, xList, yList, xList2, yList2, 'r--', xList3, yList3)

