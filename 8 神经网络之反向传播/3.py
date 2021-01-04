import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

m = 5000 # number of data
n = 400 # dimensions of input data
L = 3 # layer of neuron network
s = [0, n, 25, 10] # size of each layer
Z = []

data = loadmat('ex4data1.mat')
X = data['X']
X = np.transpose(X.reshape(m, 20, 20), [0, 2, 1]).reshape(m, 400)
y = data['y']
y[y==10] = 0
Y = np.zeros((m, s[L]))
for i in range(m):
	Y[i][y[i]] = 1

def seq(Theta):
	theta = np.array([])
	for l in range(1, L):
		theta = np.concatenate((theta, Theta[l].flatten()))
	return theta

def unseq(theta):
	Theta = [None]
	pt = 0
	for l in range(1, L):
		Theta.append(np.array(theta[pt:pt+s[l+1]*(s[l]+1)]).reshape(s[l+1], s[l]+1))
		pt += s[l+1] * (s[l]+1)
	return Theta

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forwardPropagation(Theta):
	a = [None] * (L+1)
	a[1] = np.hstack((np.ones((m, 1)), X))
	for l in range(2, L):
		a[l] = sigmoid(np.matmul(a[l-1], Theta[l-1].T))
		a[l] = np.hstack((np.ones((m, 1)), a[l]))
	a[L] = sigmoid(np.matmul(a[L-1], Theta[L-1].T))
	return a

def J(Theta, lamb):
	res = 0
	FP = forwardPropagation(Theta)
	res -= np.sum(Y * np.log(FP[L]) + (1-Y) * np.log(1-FP[L]))
	for l in range(1, L):
		res += lamb / 2 * np.sum(np.power(Theta[l][:, 1:], 2))
	res /= m

	# print(res)
	Z.append(res)

	return res

def backPropagation(Theta, lamb):
	Delta = [None] * L # (s_{l+1}, s_l+1)
	for l in range(1, L):
		Delta[l] = np.zeros((s[l+1], s[l]+1))
	delta = [None] * (L+1) # (s_l, )
	a = forwardPropagation(Theta)
	for i in range(m):
		delta[L] = a[L][i:i+1, :].T - Y[i:i+1, :].T
		for l in range(L-1, 1, -1):
			delta[l] = (np.matmul(Theta[l].T, delta[l+1]) * (a[l][i:i+1, :].T * (1 - a[l][i:i+1, :].T)))[1:, :]
		for l in range(1, L):
			Delta[l] += np.matmul(delta[l+1], a[l][i:i+1, :])
	D = [None] * L # (s_{l+1}, s_l+1)
	for l in range(1, L):
		D[l] = (Delta[l] + lamb * np.hstack((np.zeros((s[l+1], 1)), Theta[l][:, 1:]))) / m
	return D

def _J(theta, lamb):
	return J(unseq(theta), lamb)


Theta0 = [None] * L
for l in range(1, L):
	Theta0[l] = np.random.random((s[l+1], s[l]+1))
	Theta0[l] = (Theta0[l] - 0.5) / 4
res = minimize(fun = lambda theta, lamb: J(unseq(theta), lamb), 
			   x0 = seq(Theta0), 
			   args = (1), 
			   jac = lambda theta, lamb: seq(backPropagation(unseq(theta), lamb)), 
			   method = 'CG', 
			   tol = 1e-4)
print(res)
np.save('theta_CG.npy', res.x)

ax = plt.subplot(1, 1, 1)
ax.set_ylabel('Cost')
ax.plot(range(len(Z)), Z)
plt.show()