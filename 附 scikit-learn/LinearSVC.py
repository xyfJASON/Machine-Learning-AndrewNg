"""
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', *, dual=True, 
tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
class_weight=None, verbose=0, random_state=None, max_iter=1000)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import LinearSVC

data = loadmat('data/ex6data1.mat')
X = data['X']
y = data['y'].flatten()

Cs = [1, 100]
fig, ax = plt.subplots(1, 2)
fig.suptitle('LinearSVC')
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100), 
	np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
for i in range(2):
	C = Cs[i]
	clf = LinearSVC(C=C)
	clf.fit(X, y)
	print(clf.predict(X[:10]))

	ax[i].plot(X[y==0][:, 0], X[y==0][:, 1], 'x', color='red')
	ax[i].plot(X[y==1][:, 0], X[y==1][:, 1], 'o', color='blue')
	ax[i].contour(x1, x2, clf.decision_function(
		np.array([x1, x2]).transpose(1, 2, 0).reshape(10000, 2)
		).reshape(100, 100), [0])
	ax[i].set_xlabel('x1')
	ax[i].set_ylabel('x2')
	ax[i].set_title('C={0}'.format(C))
	ax[i].axis('square')

plt.show()