import sys
import numpy as np
from scipy.io import loadmat

class xyfPCA:
	"""
	My implementation of Principal Component Analysis
	"""

	def __init__(self):
		self.Xmean, self.Xstd = np.empty(0), np.empty(0)
		self.u = np.empty(0)
		self.dimension = -1
		return

	def normalization(self, X, inv=False):
		if self.Xmean.size == 0 or self.Xstd.size == 0:
			print("Should train PCA model first.")
			print("Try xyfPCA.train() method. ")
			return np.empty(0)
		return X * self.Xstd + self.Xmean if inv else (X - self.Xmean) / self.Xstd

	def transform(self, X, inv=False):
		if self.Xmean.size == 0 or self.Xstd.size == 0 or self.u.size == 0:
			print("Should train PCA model first.")
			print("Try xyfPCA.train() method. ")
			return
		if inv == False:
			return self.normalization(X, inv=False) @ self.u[:, :self.dimension]
		else:
			return self.normalization(X @ self.u[:, :self.dimension].T, inv=True)

	def train(self, X, dim=-1):
		"""
		X is the input data: (m, n)

		dim is the dimension after reduction
		if dim=-1, then the program select the smallest dim
		such that 99% of variance is retained
		
		return the data after reduction: (m, dim)
		"""
		self.Xmean = np.mean(X, axis=0)
		self.Xstd = np.std(X, axis=0, ddof=1)
		Xnorm = self.normalization(X, inv=False)
		if Xnorm.size == 0:
			return
		self.u, s, v = np.linalg.svd(Xnorm.T @ Xnorm)
		if dim == -1:
			dim = 1
			while s[:dim].sum() / s.sum() < 0.99:
				dim += 1
		self.dimension = dim
		return Xnorm @ self.u[:, :dim]
