"""
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', 
coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', 
break_ties=False, random_state=None)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

data = loadmat('data/ex6data2.mat')
X = data['X']
y = data['y'].flatten()

Cs, gammas = [1, 100], [1, 10, 30]
fig, ax = plt.subplots(2, 3)
fig.suptitle('SVC')
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
	np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
for i in range(2):
	for j in range(3):
		C, gamma = Cs[i], gammas[j]
		clf = SVC(C=C, kernel='rbf', probability=True, gamma=gamma)
		clf.fit(X, y)

		ax[i][j].plot(X[y==0][:, 0], X[y==0][:, 1], 'x', color='red', alpha=0.5)
		ax[i][j].plot(X[y==1][:, 0], X[y==1][:, 1], 'o', color='blue', markerfacecolor='none', alpha=0.5)
		ax[i][j].contour(x1, x2, clf.decision_function(
			np.array([x1, x2]).transpose(1, 2, 0).reshape(10000, 2)
			).reshape(100, 100), [0])
		ax[i][j].set_title('C={0}, gamma={1}'.format(C, gamma))

plt.show()