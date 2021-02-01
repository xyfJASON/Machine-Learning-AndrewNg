import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

lamb = 1
alpha = 0.1
iteration = 1000
m = 5000
n = 401

data = loadmat('ex3data1.mat')
X = data['X']
X = np.transpose(X.reshape((5000, 20, 20)), [0, 2, 1]).reshape(5000, 400)
X = np.hstack((np.ones((5000, 1)), X))
y = data['y']
y[y==10] = 0

def predict(Theta, x):
	return np.matmul(Theta, x).argmax(axis=0)

Theta = np.loadtxt('theta.txt', delimiter=',')
Sum = np.zeros(10)
Hit = np.zeros(10)
Accuracy = np.zeros(10)
hit = 0
for id in range(5000):
	Sum[y[id][0]] += 1
	if predict(Theta, X[id:id+1, :].T) == y[id][0]:
		Hit[y[id][0]] += 1
		hit += 1
for i in range(10):
	Accuracy[i] = Hit[i] / Sum[i]
	print(str(i)+":", str(Accuracy[i] * 100)+"%")
print("tot:", str(hit / 5000 * 100)+"%")