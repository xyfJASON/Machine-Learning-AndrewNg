import numpy as np
import matplotlib.pyplot as plt
import math

alpha = 0.01
iteration = 10000
Z = []

def Normalization(data):
	return (data - data.mean(axis = 0)) / data.std(axis = 0, ddof = 1)

def h(T, x):
	return 1 / (1 + np.e ** (-np.matmul(T.T, x)[0][0]))

def J(T, X, Y):
	res = 0
	for i in range(m):
		res -= Y[i] * math.log(h(T, X[i:i+1, :].T)) + \
		(1 - Y[i]) * math.log(1 - h(T, X[i:i+1, :].T))
	res /= m
	return res

def partJ(T, X, Y):
	res = np.zeros((n, 1))
	for i in range(m):
		res += (h(T, X[i:i+1, :].T) - Y[i]) * X[i:i+1, :].T
	res /= m
	return res

def GradientDescent(X, Y):
	T = np.zeros((n, 1))
	for t in range(iteration):
		T = T - alpha * partJ(T, X, Y)
		Z.append(J(T, X, Y))
	return T

data = np.genfromtxt("ex2data1.txt", delimiter = ',')
(m, n) = data.shape
data[:, :-1] = Normalization(data[:, :-1])
X = np.column_stack((np.ones((m, 1)), data[:, :-1]))
Y = data[:, -1]
T = GradientDescent(X, Y)
print(T)
print(J(T, X, Y))

p1 = plt.subplot(111)
p1.plot(range(1, iteration+1), Z)
p1.set_xlabel("Iteration")
p1.set_ylabel("Cost")
plt.show()