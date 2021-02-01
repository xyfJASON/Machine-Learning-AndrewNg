import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def featurePrepare(X, y):
	"""
	feature extension & normalization
	"""
	global meanX, stdX, meany, stdy
	res = np.empty((X.shape[0], 0))
	for i in range(dim):
		tmpX = X ** i
		meanX.append(np.mean(tmpX, axis=0))
		stdX.append(np.std(tmpX, axis=0))
		if i:
			tmpX = (tmpX - meanX[i]) / stdX[i] if stdX[i] else tmpX - meamX[i]
		res = np.hstack((res, tmpX))
	meany = np.mean(y, axis=0)
	stdy = np.std(y, axis=0)
	y = (y - meany) / stdy if stdy else y - meany
	return res, y

def h(theta, x):
	global meanX, stdX, meany, stdy
	y_h = 0
	for i in range(dim):
		if i:
			y_h += theta[i] * ((x ** i - meanX[i]) / stdX[i])
		else:
			y_h += theta[i]
	y_h = y_h * stdy + meany
	return y_h

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

def Train(X, y, lamb):
	return minimize(fun = lambda theta, X, y, lamb: J(unseq(theta), X, y, lamb), 
					x0 = np.ones((dim, )), 
					jac = lambda theta, X, y, lamb: seq(partJ(unseq(theta), X, y, lamb)), 
					args = (X, y, lamb), 
					method = 'CG')

data = loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = \
data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']

dim = 9
meanX, stdX = [], []
meany, stdy = 0, 0

lamb_range = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
Z_train, Z_valid = [], []
for lamb in lamb_range:
	polyXval, polyyval = featurePrepare(Xval, yval)
	polyX, polyy = featurePrepare(X, y)
	res = Train(polyX, polyy, lamb=lamb)
	Z_train.append(J(unseq(res.x), polyX, polyy, lamb=0))
	Z_valid.append(J(unseq(res.x), polyXval, polyyval, lamb=0))

	polyXtest, polyytest = featurePrepare(Xtest, ytest)
	print(lamb, J(unseq(res.x), polyXtest, polyytest, lamb=0))

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('lambda')
ax.set_ylabel('Cost')
ax.plot(lamb_range, Z_train, color='darkviolet', label='Train')
ax.plot(lamb_range, Z_valid, color='tomato', label='Validation')
ax.legend()
plt.show()