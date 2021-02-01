import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

alpha = 0.1
iteration = 100
lamb = 1

m = 5000 # number of data
n = 400 # dimensions of input data
L = 3 # layer of neuron network
s = [0, n, 25, 10] # size of each layer

data = loadmat('ex4data1.mat')
X = data['X']
X = np.transpose(X.reshape(m, 20, 20), [0, 2, 1]).reshape(m, 400)
y = data['y']
y[y==10] = 0
Y = np.zeros((m, s[L]))
for i in range(m):
	Y[i][y[i]] = 1

def unseq(theta):
	Theta = [None]
	pt = 0
	for l in range(1, L):
		Theta.append(np.array(theta[pt:pt+s[l+1]*(s[l]+1)]).reshape(s[l+1], s[l]+1))
		pt += s[l+1] * (s[l]+1)
	return Theta

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forwardPropagation(x):
	"""
	x: (n, 1)
	"""
	a = [None] * (L+1)
	a[1] = np.vstack((np.ones((1, 1)), x))
	for l in range(2, L):
		a[l] = sigmoid(np.matmul(Theta[l-1], a[l-1]))
		a[l] = np.vstack((np.ones((1, 1)), a[l]))
	a[L] = sigmoid(np.matmul(Theta[L-1], a[L-1]))
	return a

def predict(x):
	"""
	x: (n, 1)
	"""
	return forwardPropagation(x)[L].argmax(axis=0)

Theta = unseq(np.load('theta_CG.npy', allow_pickle=True))
Sum = np.zeros(10)
Hit = np.zeros(10)
Accuracy = np.zeros(10)
hit = 0
for id in range(m):
	Sum[y[id][0]] += 1
	if predict(X[id:id+1, :].T) == y[id][0]:
		Hit[y[id][0]] += 1
		hit += 1
for i in range(10):
	Accuracy[i] = Hit[i] / Sum[i]
	print(str(i)+":", str(Accuracy[i] * 100)+"%")
print("tot:", str(hit / 5000 * 100)+"%")
