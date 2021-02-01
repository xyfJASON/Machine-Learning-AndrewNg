"""
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, 
tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
class_weight=None, random_state=None, solver='lbfgs', max_iter=100, 
multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

class sklearn.preprocessing.PolynomialFeatures(degree=2, *, 
interaction_only=False, include_bias=True, order='C')
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt('data/ex2data2.txt', delimiter=',')
y = data[:, -1]
X = data[:, :-1]
poly = PolynomialFeatures(degree=5)
newX = poly.fit_transform(X)

fig, ax = plt.subplots(1, 4)
fig.suptitle('Logistic Regression')
Cs = [0.01, 1, 300, -1]
for i in range(4):
	C = Cs[i]
	if i < 3:
		clf = LogisticRegression(C=C, max_iter=1000)
	else:
		clf = LogisticRegressionCV(max_iter=1000)
	clf.fit(newX, y)
	print(clf.score(newX, y))

	if i < 3:
		ax[i].set_title('C={0}'.format(C))
	else:
		ax[i].set_title('LogisticRegressionCV')
	ax[i].plot(X[y==0][:, 0], X[y==0][:, 1], 'x', color='red')
	ax[i].plot(X[y==1][:, 0], X[y==1][:, 1], 'o', color='blue', markerfacecolor='none')
	x1, x2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
		np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
	ax[i].contour(x1, x2, clf.predict_proba(
		poly.transform(np.array([x1, x2]).transpose(1, 2, 0).reshape((10000, 2)))
		)[:, 0].reshape(100, 100), [0.5])
	ax[i].axis('square')

plt.show()
