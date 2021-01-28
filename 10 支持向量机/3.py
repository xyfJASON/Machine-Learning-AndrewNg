import numpy as np
from scipy.io import loadmat
from sklearn import svm
import matplotlib.pyplot as plt

data = loadmat('ex6data3.mat')
X = data['X']
y = data['y'].flatten()

maxscore, maxC, maxgamma = 0, 0, 0
for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
	for gamma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
		clf = svm.SVC(C=C, kernel='rbf', gamma=gamma, probability=True)
		clf.fit(X, y)
		score = clf.score(X, y)
		print(C, gamma, score)
		if score > maxscore:
			maxscore, maxC, maxgamma = score, C, gamma

clf = svm.SVC(C=maxC, kernel='rbf', gamma=maxgamma, probability=True)
clf.fit(X, y)

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('C={0}, gamma={1}'.format(maxC, maxgamma))
ax.plot(X[(y==1).flatten()][:, 0], X[(y==1).flatten()][:, 1], '+', color='black')
ax.plot(X[(y==0).flatten()][:, 0], X[(y==0).flatten()][:, 1], 'o', color='gold')

def calc(x1, x2):
	res = np.empty(x1.shape)
	for i in range(x1.shape[0]):
		for j in range(x1.shape[1]):
			res[i, j] = clf.decision_function(np.array([[x1[i, j], x2[i, j]]]))
	return res

x1, x2 = np.meshgrid(np.arange(-0.7, 0.4, 0.01), np.arange(-0.7, 0.6, 0.01))
ax.contour(x1, x2, calc(x1, x2), [0])

plt.show()