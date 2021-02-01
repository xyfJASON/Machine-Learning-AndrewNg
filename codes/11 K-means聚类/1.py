import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

X = loadmat('ex7data2.mat')['X']

def J(X, cluster, centroid):
	res = 0
	for i in range(X.shape[0]):
		res += np.dot(X[i]-centroid[cluster[i]], X[i]-centroid[cluster[i]])
	return res

def K_means(K, X, iteration=100):
	(m, n) = X.shape
	bestCluster, bestCentroid, bestJ = np.empty(m), np.empty((K, n)), np.inf
	for iter in range(iteration):
		centroid = X[np.random.randint(0, m, K)]
		cluster = np.empty(m, dtype='int')
		while True:
			ncentroid = np.zeros((K, n))
			cnt = np.zeros(K)
			for i in range(m):
				cluster[i] = np.argmin(np.sum((X[i]-centroid)**2, axis=1))
				ncentroid[cluster[i]] += X[i]
				cnt[cluster[i]] += 1
			ncentroid[cnt!=0] /= cnt[cnt!=0][:, np.newaxis]
			ncentroid[cnt==0] = X[np.random.randint(0, m, len(cnt[cnt==0]))]
			if (centroid == ncentroid).all():
				break
			centroid = ncentroid.copy()
		cost = J(X, cluster, centroid)
		if cost < bestJ:
			bestCluster, bestCentroid, bestJ = cluster.copy(), centroid.copy(), cost.copy()
	return bestCluster, bestCentroid

cl, ce = K_means(3, X, iteration=100)

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.plot(X[cl==0][:, 0], X[cl==0][:, 1], 'o', color='blue', markerfacecolor='none', alpha=0.4)
ax.plot(X[cl==1][:, 0], X[cl==1][:, 1], 'o', color='green', markerfacecolor='none', alpha=0.4)
ax.plot(X[cl==2][:, 0], X[cl==2][:, 1], 'o', color='red', markerfacecolor='none', alpha=0.4)
ax.plot(ce[0, 0], ce[0, 1], '*', color='blue', ms=10)
ax.plot(ce[1, 0], ce[1, 1], '*', color='green', ms=10)
ax.plot(ce[2, 0], ce[2, 1], '*', color='red', ms=10)
ax.plot([], [], '*', color='black', ms=10, label='cluster centroid')
ax.legend()

plt.show()
