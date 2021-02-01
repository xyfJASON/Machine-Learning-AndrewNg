import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

def J(Y, R, X, Theta, lamb):
	return 0.5 * (np.sum(((X @ Theta.T - Y) * R) ** 2) + \
		lamb * np.sum(X ** 2) + lamb * np.sum(Theta ** 2))

def partJ(Y, R, X, Theta, lamb):
	return np.concatenate((
		( ((X @ Theta.T - Y) * R) @ Theta + lamb * X ).flatten(), 
		( ((X @ Theta.T - Y) * R).T @ X + lamb * Theta ).flatten()
	))

def train(Y, R, lamb, n):
	(n_m, n_u) = Y.shape
	xt = np.empty(n*n_m+n*n_u)
	return minimize(fun = lambda xt, Y, R, lamb : J(Y, R, xt[:n*n_m].reshape((n_m, n)), \
													xt[n*n_m:].reshape((n_u, n)), lamb), 
					x0 = np.random.randn(n*n_m+n*n_u), 
					args = (Y, R, lamb), 
					method = 'TNC', 
					jac = lambda xt, Y, R, lamb: partJ(Y, R, xt[:n*n_m].reshape((n_m, n)), \
													xt[n*n_m:].reshape((n_u, n)), lamb)
					)

def predict(Y, R, xt, n):
	(n_m, n_u) = Y.shape
	X, Theta = xt[:n*n_m].reshape((n_m, n)), xt[n*n_m:].reshape((n_u, n))
	return X @ Theta.T

data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
Ymean = Y.mean(axis=1, keepdims=True)
# res = train(Y-Ymean, R, lamb=1, n=50)
# print(res)
# np.save('xt.npy', res.x)
# xt = res.x

xt = np.load('xt.npy')
pred = predict(Y, R, xt, n=50) + Ymean
movie_list = []
with open('movie_ids.txt', encoding='latin-1') as file:
	for line in file:
		movie_list.append(' '.join(line.strip().split(' ')[1: ]))
movie_list = np.array(movie_list)
idx = np.argsort(pred[:, 0])[::-1]
print('Top 10 movies for user 1:')
print(pred[:, 0][idx][:10])
for movie in movie_list[idx][:10]:
	print(movie)
