import numpy as np

def J(T, X, Y):
	res = 0
	for i in range(m):
		res += (np.matmul(T.T, X[i:i+1, :].T) - Y[i:i+1, :]) ** 2
	res /= 2 * m;
	return res

data = np.genfromtxt("ex1data2.txt", delimiter = ',')
(m, n) = data.shape
X = np.column_stack((np.ones((m, 1)), data[:, :-1]))
Y = data[:, -1:]
T = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), Y)
print(T)
print(J(T, X, Y))