import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01
lamb = 5
iteration = 1000000
Z = []

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

def GradientDescent(X, Y):
	T = np.zeros((n, 1))
	for t in range(iteration):
		T = T - alpha * partJ_reg(T, X, Y)
		Z.append(J_reg(T, X, Y))
	return T

def extend(x1, x2):
	X = []
	for j in range(4):
		for k in range(4):
			X.append(np.power(x1, j) * np.power(x2, k))
	return X

data = np.genfromtxt("ex2data2.txt", delimiter = ',')
(m, n) = data.shape
Y = data[:, -1:]
n = 16
X = np.zeros((m, n))
for i in range(m):
	X[i] = extend(data[i][0], data[i][1])

T = GradientDescent(X, Y)
print(T.T)
print(J_reg(T, X, Y))

#============================== draw the picture ==============================#

def calc(x1, x2):
	res = np.zeros((x1.shape[0], x1.shape[1]))
	for ix in range(x1.shape[0]):
		for iy in range(x1.shape[1]):
			tmp = np.array(extend(x1[ix][iy], x2[ix][iy]), ndmin = 2)
			res[ix][iy] = h(T, tmp.T)
	return res

minx1, maxx1 = data[:, 0].min(), data[:, 0].max()+0.1
minx2, maxx2 = data[:, 1].min(), data[:, 1].max()+0.1
delta = 0.01
x1, x2 = np.meshgrid(np.arange(minx1, maxx1, delta), np.arange(minx2, maxx2, delta))

p1 = plt.subplot(121)
plt.contour(x1, x2, calc(x1, x2), [0.5], colors = 'magenta')

X0 = data[data[:, -1] == 0, :]
X1 = data[data[:, -1] == 1, :]
p1.set_title("lambda = " + str(lamb))
p1.set_xlabel("Microchip test 1")
p1.set_ylabel("Microchip test 2")
p1.scatter(X0[:, 0:1], X0[:, 1:2], marker = 'x', c = 'r')
p1.scatter(X1[:, 0:1], X1[:, 1:2], marker = 'o')

p2 = plt.subplot(122)
p2.plot(range(1, iteration+1), Z)
p2.set_title("lambda = " + str(lamb))
p2.set_xlabel("Iteration")
p2.set_ylabel("Cost")

plt.show()