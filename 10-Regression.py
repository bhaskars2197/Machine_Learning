from math import ceil
import numpy as np
from scipy import linalg

def lowess(x, y, tau = 0.25):
    m = len(x)
    yest = np.zeros(n)
    
    #Initializing all weights from the bell shape kernel function    
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(m)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest
n = 100
x = np.linspace(0, 2 * 22/7.0, n)
print("==========================values of x=====================")
print(x)
y = np.sin(x) + 0.3*np.random.randn(n)
print("================================Values of y===================")
print(y)
f = 0.25
yest = lowess(x, y)
import matplotlib.pyplot as plt
plt.clf()
plt.plot(x, y, label='y noise')
plt.plot(x, yest, label='y pred')
plt.legend()
plt.show()
