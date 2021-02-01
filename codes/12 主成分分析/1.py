import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def PCA(X, dim = -1):
	"""
	X is the input data: (m, n)

	dim is the dimension after reduction
	if dim=-1, then the program select the smallest dim
	such that 99% of variance is retained
	
	return the data after reduction: (m, dim)
	and the data recovered from reduced data: (m, n)
	"""
	Xmean = np.empty((1, X.shape[1]))
	Xstd = np.empty((1, X.shape[1]))
	def normalization(X, k):
		global Xmean, Xstd
		if k == 1:
			Xmean = np.mean(X, axis=0)
			Xstd = np.std(X, axis=0, ddof=1)
			return (X - Xmean) / Xstd
		else:
			return X * Xstd + Xmean

	Xnorm = normalization(X, 1)
	u, s, v = np.linalg.svd(Xnorm.T @ Xnorm)
	if dim == -1:
		dim = 1
		while s[:dim].sum() / s.sum() < 0.99:
			dim += 1
	return Xnorm @ u[:, :dim], normalization(Xnorm @ u[:, :dim] @ u[:, :dim].T, 0)

X = loadmat('ex7data1.mat')['X']
redX, recX = PCA(X, dim=1)

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.plot(X[:, 0], X[:, 1], 'o', color='black', markerfacecolor='none')

ax.plot(recX[:, 0], recX[:, 1], 'o', color='red', markerfacecolor='none')
ax.axis('square')

plt.show()
