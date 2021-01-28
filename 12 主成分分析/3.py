import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

X = loadmat('ex7faces.mat')['X']
X = np.transpose(X.reshape((5000, 32, 32)), [0, 2, 1]).reshape(5000, 1024)
X = -X
dim = 4

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
			print(s[:dim].sum() / s.sum())
			dim += 1
	return Xnorm @ u[:, :dim], normalization(Xnorm @ u[:, :dim] @ u[:, :dim].T, 0)

def show_a_face(face, ax):
	"""
	face.shape: (1024, )
	"""
	ax.matshow(face.reshape((32, 32)), cmap=matplotlib.cm.binary)
	ax.axis('off')

redX, recX = PCA(X, dim=dim)

fig, ax = plt.subplots(10, 10)
fig.suptitle('dim={0}'.format(dim))
for i in range(10):
	for j in range(10):
		show_a_face(recX[i*10+j, :], ax[i][j])
plt.show()
