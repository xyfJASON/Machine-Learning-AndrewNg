"""
class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, normalize=False, 
copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)

class sklearn.preprocessing.PolynomialFeatures(degree=2, *, 
interaction_only=False, include_bias=True, order='C')
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

data = loadmat('data/ex5data1.mat')
X = data['X']
y = data['y']
poly = PolynomialFeatures(degree=8)
newX = poly.fit_transform(X)

alphas = [0, 0.05, 10, -1]
fig, ax = plt.subplots(1, 4)
fig.suptitle('Polynomial Regression')
for i in range(4):
	alpha = alphas[i]
	if alpha == -1:
		reg = RidgeCV(normalize=True)
	else:
		reg = Ridge(alpha=alpha, normalize=True)
	reg.fit(newX, y)
	print(reg.score(newX, y))

	if alpha >= 0:
		ax[i].set_title('alpha={0}'.format(alpha))
	else:
		ax[i].set_title('RidgeCV')
	ax[i].plot(X.flatten(), y, 'x', color='red')
	ax[i].plot(np.linspace(X.min()-5, X.max()+5, 100), \
		reg.predict(poly.transform( np.linspace(X.min()-5, X.max()+5, 100).reshape((100, 1))) ))

plt.show()
