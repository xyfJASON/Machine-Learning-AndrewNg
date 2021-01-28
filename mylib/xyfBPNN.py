import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class xyfBPNN:
	"""
	My implementation of Back Propagation Neural Network
	"""

	def __init__(self, nData, nLayer, lSizes, regLambda=0, method=None, tol=None, show=True):
		self.m = nData
		self.L = nLayer
		self.s = np.concatenate((np.zeros(1, dtype = 'int'), lSizes))
		self.lamb = regLambda
		self.method = method
		self.tol = tol
		self.show = show
		self.param = None
		self.Z = []

	def seq(self, Theta):
		theta = np.array([])
		for l in range(1, self.L):
			theta = np.concatenate((theta, Theta[l].flatten()))
		return theta

	def unseq(self, theta):
		Theta = [None]
		pt = 0
		for l in range(1, self.L):
			Theta.append(np.array(theta[pt:pt+self.s[l+1]*(self.s[l]+1)]).reshape(self.s[l+1], self.s[l]+1))
			pt += self.s[l+1] * (self.s[l]+1)
		return Theta

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def forwardPropagation(self, Theta, X):
		a = [None] * (self.L+1)
		self.m = X.shape[0]
		a[1] = np.hstack((np.ones((self.m, 1)), X))
		for l in range(2, self.L):
			a[l] = self.sigmoid(np.matmul(a[l-1], Theta[l-1].T))
			a[l] = np.hstack((np.ones((self.m, 1)), a[l]))
		a[self.L] = self.sigmoid(np.matmul(a[self.L-1], Theta[self.L-1].T))
		return a

	def J(self, Theta, lamb, X, Y):
		res = 0
		FP = self.forwardPropagation(Theta, X)
		res -= np.sum(Y * np.log(FP[self.L]) + (1-Y) * np.log(1-FP[self.L]))
		for l in range(1, self.L):
			res += lamb / 2 * np.sum(np.power(Theta[l][:, 1:], 2))
		res /= self.m

		if self.show and len(self.Z) % 20 == 0:
			print(res)
		self.Z.append(res)

		return res

	def backPropagation(self, Theta, lamb, X, Y):
		Delta = [None] * self.L # (s_{l+1}, s_l+1)
		for l in range(1, self.L):
			Delta[l] = np.zeros((self.s[l+1], self.s[l]+1))
		delta = [None] * (self.L+1) # (s_l, )
		a = self.forwardPropagation(Theta, X)
		for i in range(self.m):
			delta[self.L] = a[self.L][i:i+1, :].T - Y[i:i+1, :].T
			for l in range(self.L-1, 1, -1):
				delta[l] = (np.matmul(Theta[l].T, delta[l+1]) * (a[l][i:i+1, :].T * (1 - a[l][i:i+1, :].T)))[1:, :]
			for l in range(1, self.L):
				Delta[l] += np.matmul(delta[l+1], a[l][i:i+1, :])
		D = [None] * self.L # (s_{l+1}, s_l+1)
		for l in range(1, self.L):
			D[l] = (Delta[l] + lamb * np.hstack((np.zeros((self.s[l+1], 1)), Theta[l][:, 1:]))) / self.m
		return D

	def train(self, X, Y):
		Theta0 = [None] * self.L
		for l in range(1, self.L):
			Theta0[l] = np.random.random((self.s[l+1], self.s[l]+1))
			Theta0[l] = (Theta0[l] - 0.5) / 4
		res = minimize(fun = lambda theta, lamb, X, Y: self.J(self.unseq(theta), lamb, X, Y), 
					   x0 = self.seq(Theta0), 
					   args = (self.lamb, X, Y), 
					   jac = lambda theta, lamb, X, Y: self.seq(self.backPropagation(self.unseq(theta), lamb, X, Y)), 
					   method = self.method, 
					   tol = self.tol)
		np.save('theta_' + self.method + '_' + str(self.tol) + '.npy', res.x)
		self.param = res.x
		if self.show:
			print(res)
			ax = plt.subplot(1, 1, 1)
			ax.set_xlabel('Iteration')
			ax.set_ylabel('Cost')
			ax.plot(range(len(self.Z)), self.Z)
			plt.show()
		return res

	def predict(self, inData):
		return self.forwardPropagation(self.unseq(self.param), inData)

	def classifyPredict(self, inData):
		a = self.predict(inData)
		res = []
		for i in range(inData.shape[0]):
			res.append(a[self.L][i].argmax())
		return res

	def classifyTest(self, inData, outData):
		res = self.classifyPredict(inData)
		confusion = np.zeros((self.s[self.L], self.s[self.L]))
		for i in range(inData.shape[0]):
			confusion[outData[i], res[i]] += 1
		accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
		precision = np.zeros(self.s[self.L])
		recall = np.zeros(self.s[self.L])
		for i in range(self.s[self.L]):
			precision[i] = confusion[i, i] / np.sum(confusion[:, i])
			recall[i] = confusion[i, i] / np.sum(confusion[i, :])
		return accuracy, precision, recall
