import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

data = loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

Xmeans = X.mean(axis=0)
Xcov = ((X - Xmeans).T @ (X - Xmeans)) / X.shape[0]
normDist = multivariate_normal(mean=Xmeans, cov=Xcov)

x1, x2 = np.meshgrid(np.linspace(0, 30, 100), np.linspace(0, 30, 100))

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Throughput (mb/s)')
ax.plot(X[:, 0], X[:, 1], 'x', color='blue', alpha=0.3)

ax.contour(x1, x2, 
	normDist.pdf(np.array([x1, x2]).transpose(1, 2, 0).reshape(10000, 2)).reshape(100, 100), 
	levels=[1e-22, 1e-15, 1e-10, 1e-5, 1e-2], colors=plt.cm.tab20c(np.linspace(0, 1, 12)))

plt.show()
