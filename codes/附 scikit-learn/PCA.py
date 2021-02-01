"""
class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, 
svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = loadmat('data/ex7data1.mat')
X = data['X']

pca = PCA(n_components=1)
pca.fit(X)
redX = pca.transform(X) # reduced X
recX = pca.inverse_transform(redX) # recovered X

ax = plt.subplot(1, 1, 1)
ax.plot(X[:, 0], X[:, 1], 'o', color='black', markerfacecolor='none')
ax.plot(recX[:, 0], recX[:, 1], 'o', color='red', markerfacecolor='none')
ax.set_title('PCA')
ax.axis('square')
plt.show()
