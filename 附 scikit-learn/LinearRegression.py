"""
class sklearn.linear_model.LinearRegression(*, fit_intercept=True, 
normalize=False, copy_X=True, n_jobs=None, positive=False)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.loadtxt('data/ex1data1.txt', delimiter=',')
y = data[:, -1]
X = data[:, :-1]

reg = LinearRegression(normalize=True)
reg.fit(X, y)
print(reg.score(X, y))

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression')
ax.plot(X.flatten(), y, 'x', color='red')
ax.plot(X, reg.predict(X))
plt.show()
