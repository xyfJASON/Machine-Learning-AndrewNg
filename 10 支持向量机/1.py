import numpy as np
from scipy.io import loadmat
from sklearn import svm
import matplotlib.pyplot as plt

data = loadmat('ex6data1.mat')
X = data['X']
y = data['y'].flatten()

clf = svm.SVC(C=1, kernel='linear')
clf.fit(X, y)

ax = plt.subplot(1, 2, 1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('C=1')
ax.plot(X[(y==1).flatten()][:, 0], X[(y==1).flatten()][:, 1], '+', color='black')
ax.plot(X[(y==0).flatten()][:, 0], X[(y==0).flatten()][:, 1], 'o', color='gold')

def calc(x1, x2):
	res = np.empty(x1.shape)
	for i in range(x1.shape[0]):
		for j in range(x1.shape[1]):
			res[i, j] = clf.decision_function(np.array([[x1[i, j], x2[i, j]]]))
	return res

x1, x2 = np.meshgrid(np.arange(0, 4.5, 0.05), np.arange(1.3, 4.5, 0.05))
ax.contour(x1, x2, calc(x1, x2), [0])

clf = svm.SVC(C=100, kernel='linear')
clf.fit(X, y)

ax = plt.subplot(1, 2, 2)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('C=100')
ax.plot(X[(y==1).flatten()][:, 0], X[(y==1).flatten()][:, 1], '+', color='black')
ax.plot(X[(y==0).flatten()][:, 0], X[(y==0).flatten()][:, 1], 'o', color='gold')
ax.contour(x1, x2, calc(x1, x2), [0])

plt.show()
