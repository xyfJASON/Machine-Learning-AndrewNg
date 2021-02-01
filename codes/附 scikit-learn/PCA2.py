"""
class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, 
svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

X = loadmat('data/ex7faces.mat')['X']
X = np.transpose(X.reshape((5000, 32, 32)), [0, 2, 1]).reshape(5000, 1024)
X = -X

pca = PCA(n_components=0.99)
pca.fit(X)
redX = pca.transform(X) # reduced X
recX = pca.inverse_transform(redX) # recovered X

def show_a_face(face, ax):
	"""
	face.shape: (1024, )
	"""
	ax.matshow(face.reshape((32, 32)), cmap=matplotlib.cm.binary)
	ax.axis('off')

fig, ax = plt.subplots(5)
fig.suptitle('99% variance', fontweight='bold')
fig.subplots_adjust(hspace=0, wspace=None)
for i in range(5):
	show_a_face(recX[i], ax[i])
plt.show()
