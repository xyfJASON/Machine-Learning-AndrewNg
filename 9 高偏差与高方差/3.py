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

theta_train = None
Z_train = []
Z_valid = []
polyXval, polyyval = featurePrepare(Xval, yval)
for i in range(1, 1+X.shape[0]):
	polyX, polyy = featurePrepare(X[:i, :], y[:i, :])
	res = Train(polyX, polyy, lamb=1)
	Z_train.append(J(unseq(res.x), polyX, polyy, lamb=0))
	Z_valid.append(J(unseq(res.x), polyXval, polyyval, lamb=0))
	if i == X.shape[0]:
		theta_train = res.x

ax1 = plt.subplot(1, 2, 1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Polynomial regression, lambda=1')
ax1.plot(X, y, 'x', color='magenta')
ax1.plot(range(-60, 50), [h(theta_train, i) for i in range(-60, 50)], '--')

ax2 = plt.subplot(1, 2, 2)
ax2.set_xlabel('size of training set')
ax2.set_ylabel('Cost')
ax2.set_title('learning curves')
ax2.plot(range(1, 1+len(Z_train)), Z_train, color='darkviolet', label='Train')
ax2.plot(range(1, 1+len(Z_valid)), Z_valid, color='tomato', label='Validation')
ax2.legend()
plt.show()