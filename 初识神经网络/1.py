import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

m = 5000
n = 401

data = loadmat('ex3data1.mat')
X = data['X'] # (5000, 400)
X = np.hstack((np.ones((m, 1)), X)) # (5000, 401)
y = data['y'] # (5000, 1)

data = loadmat('ex3weights.mat')
Theta1 = data['Theta1'] # (25, 401)
Theta2 = data['Theta2'] # (10, 26)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(x):
	"""
	x: (401, )
	"""
	a1 = x.reshape((401, 1))
	a2 = np.matmul(Theta1, a1)
	a2 = sigmoid(a2) # (25, 1)
	a2 = np.vstack((np.ones((1, 1)), a2)) # (26, 1)
	a3 = np.matmul(Theta2, a2)
	a3 = sigmoid(a3)
	return a3.argmax(axis=0)[0] + 1

Sum = np.zeros(11)
Hit = np.zeros(11)
Accuracy = np.zeros(11)
hit = 0
for id in range(5000):
	Sum[y[id][0]] += 1
	if predict(X[id, :].T) == y[id][0]:
		Hit[y[id][0]] += 1
		hit += 1
for i in range(1, 11):
	Accuracy[i] = Hit[i] / Sum[i]
	print(str(i)+":", str(Accuracy[i] * 100)+"%")
print("tot:", str(hit / 5000 * 100)+"%")