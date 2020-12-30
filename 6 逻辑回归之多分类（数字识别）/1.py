import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

lamb = 1
alpha = 0.1
iteration = 10000
m = 5000
n = 401

data = loadmat('ex3data1.mat')
X = data['X']
X = np.transpose(X.reshape((m, 20, 20)), [0, 2, 1]).reshape(m, 400)
X = np.hstack((np.ones((m, 1)), X))
y = data['y']
y[y==10] = 0

def h(Theta, x):
	return 1 / (1 + np.exp(-np.matmul(Theta.T, x)[0][0]))

def h_(Theta, X):
	return 1 / (1 + np.exp(-np.matmul(X, Theta)))

def J_reg(Theta, X, y):
	tmp = h_(Theta, X)
	return (-(np.matmul(y.T, np.log(tmp)) + np.matmul(1 - y.T, np.log(1 - tmp))) / m\
		 + lamb / 2 / m * np.matmul(Theta.T, Theta))[0][0]

def partJ_reg(Theta, X, y):
	return (np.matmul(X.T, h_(Theta, X) - y) + lamb * np.vstack((np.zeros((1, 1)), Theta[1:, :]))) / m

def GradientDescent(X, y):
	Theta = np.zeros((n, 1))
	for t in range(iteration):
		Theta = Theta - alpha * partJ_reg(Theta, X, y)
	return Theta

def classify(k):
	y_ = y.copy()
	y_[y==k] = 1
	y_[y!=k] = 0
	Theta = GradientDescent(X, y_)
	return Theta

Theta = np.zeros((10, n))
for k in range(10):
	Theta[k, :] = classify(k).flatten()
	print("Trained for:", k)

np.savetxt('theta.txt', Theta, delimiter=',')