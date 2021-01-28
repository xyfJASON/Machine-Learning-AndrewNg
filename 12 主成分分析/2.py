import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

X = loadmat('ex7faces.mat')['X']
X = np.transpose(X.reshape((5000, 32, 32)), [0, 2, 1]).reshape(5000, 1024)
X = -X

def show_a_face(face, ax):
	"""
	face.shape: (1024, )
	"""
	ax.matshow(face.reshape((32, 32)), cmap=matplotlib.cm.binary)
	ax.axis('off')

fig, ax = plt.subplots(10, 10)
for i in range(10):
	for j in range(10):
		show_a_face(X[i*10+j, :], ax[i][j])
plt.show()
