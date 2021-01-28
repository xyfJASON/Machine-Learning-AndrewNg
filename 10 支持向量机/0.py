import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data1, data2, data3 = loadmat('ex6data1.mat'), loadmat('ex6data2'), loadmat('ex6data3')

# X = data1['X']
# y = data1['y']
# X = data2['X']
# y = data2['y']
X = data3['X']
y = data3['y']
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.plot(X[(y==1).flatten()][:, 0], X[(y==1).flatten()][:, 1], '+', color='black')
ax.plot(X[(y==0).flatten()][:, 0], X[(y==0).flatten()][:, 1], 'o', color='gold')
plt.show()
