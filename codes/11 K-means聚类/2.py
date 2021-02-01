import numpy as np
from PIL import Image

def readin():
	data = np.array(Image.open('bird_small.png'))
	data = data.reshape((128*128, 3))
	return data

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

X = readin()
cl, ce = K_means(16, X, iteration=20)

comImg = np.empty((128*128, 3), dtype='uint8')
for i in range(128*128):
	comImg[i] = np.floor(ce[cl[i]])
comImg = comImg.reshape((128, 128, 3))
im = Image.fromarray(comImg)
im.save('bird_compression.png')
