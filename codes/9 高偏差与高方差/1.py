import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = \
data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']
X = np.hstack((np.ones((X.shape[0], 1)), X))
Xval = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
Xtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

def unseq(theta):
	return theta.reshape(theta.shape[0], 1)

def seq(theta):
	return theta.flatten()

def J(theta, X, y, lamb):
	m = X.shape[0]
	thetahat = np.vstack((np.zeros((1, 1)), theta[1:, :]))
	return ((theta.T@X.T@X@theta-2*theta.T@X.T@y+y.T@y+lamb*thetahat.T@thetahat)/m/2)[0][0]

def partJ(theta, X, y, lamb):
	m = X.shape[0]
	thetahat = np.vstack((np.zeros((1, 1)), theta[1:, :]))
	return (X.T@X@theta-X.T@y+lamb*thetahat)/m

def Train(X, y):
	return minimize(fun = lambda theta, X, y, lamb: J(unseq(theta), X, y, lamb), 
				   x0 = np.array([1, 1]), 
				   jac = lambda theta, X, y, lamb: seq(partJ(unseq(theta), X, y, lamb)), 
				   args = (X, y, 1), 
				   method = 'CG')

res = Train(X, y)
print(res)

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
rng = np.arange(np.min(X[:, 1]), np.max(X[:, 1]))
ax.plot(X[:, 1], y, 'o', color='magenta')
ax.plot(rng, res.x[0]+res.x[1]*rng)
plt.show()