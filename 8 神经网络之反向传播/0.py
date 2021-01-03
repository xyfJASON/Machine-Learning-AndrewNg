import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

lamb = 1

m = 5000
n = 400
L = 3
s = [0, n, 25, 10] # size of each layer

data = loadmat('ex4data1.mat')
X = data['X'] # (5000, 400)
y = data['y'] # (5000, 1)
Y = np.zeros((m, s[L]))
for i in range(m):
	Y[i][y[i]-1] = 1

data = loadmat('ex4weights.mat')
Theta = [None] * 3
Theta[1] = data['Theta1'] # (25, 401)
Theta[2] = data['Theta2'] # (10, 26)

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

def J(reg):
	res = 0
	for i in range(m):
		tmp = forwardPropagation(X[i:i+1, :].T)
		res -= np.sum(Y[i:i+1, :].T * np.log(tmp[L]))
		res -= np.sum((1-Y[i:i+1, :].T) * np.log(1-tmp[L]))
	if reg:
		for l in range(1, L):
			res += lamb / 2 * np.sum(np.power(Theta[l], 2))
	res /= m
	return res

def predict(x):
	"""
	x: (n, 1)
	"""
	return forwardPropagation(x)[L].argmax(axis=0) + 1

print(Theta[1])
print(J(reg=True))
Sum = np.zeros(11)
Hit = np.zeros(11)
Accuracy = np.zeros(11)
hit = 0
for id in range(5000):
	Sum[y[id][0]] += 1
	if predict(X[id:id+1, :].T) == y[id][0]:
		Hit[y[id][0]] += 1
		hit += 1
for i in range(1, 11):
	Accuracy[i] = Hit[i] / Sum[i]
	print(str(i)+":", str(Accuracy[i] * 100)+"%")
print("tot:", str(hit / 5000 * 100)+"%")