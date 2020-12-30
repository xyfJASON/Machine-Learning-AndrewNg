import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

data = loadmat('ex3data1.mat')
X = data['X']
X = np.transpose(X.reshape((5000, 20, 20)), [0, 2, 1]).reshape(5000, 400)
y = data['y']
y[y==10] = 0

def show_a_number(num, ax):
	"""
	num.shape: (400, )
	"""
	ax.matshow(num.reshape((20, 20)), cmap=matplotlib.cm.binary)
	ax.axis('off')

fig, ax = plt.subplots(10, 10)
for i in range(10):
	for j in range(10):
		id = np.random.randint(0, 5000)
		show_a_number(X[id, :], ax[i][j])
plt.show()