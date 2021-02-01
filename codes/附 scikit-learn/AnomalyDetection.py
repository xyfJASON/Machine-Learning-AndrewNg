"""
class sklearn.covariance.EllipticEnvelope(*, store_precision=True, assume_centered=False, 
support_fraction=None, contamination=0.1, random_state=None)

sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, 
average='binary', sample_weight=None, zero_division='warn')
"""

import numpy as np
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

data = loadmat('data/ex8data1.mat')
X, Xval, yval = data['X'], data['Xval'], data['yval'].flatten()

bestc, bestf1 = 0, 0
for c in np.linspace(0, 0.1, 100):
	ano = EllipticEnvelope(contamination=c)
	ano.fit(X)
	pred = ano.predict(X)
	pred[pred==1] = 0
	pred[pred==-1] = 1
	f1 = f1_score(yval, pred)
	if f1 > bestf1:
		bestc, bestf1 = c, f1

ano = EllipticEnvelope(contamination=bestc)
print(bestc)
ano.fit(X)
pred = ano.predict(X)

ax = plt.subplot(1, 1, 1)
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Throughput (mb/s)')
ax.plot(X[:, 0], X[:, 1], 'x', alpha=0.5, color='blue')
ax.plot(X[pred==-1][:, 0], X[pred==-1][:, 1], 'o', color='red', markerfacecolor='none', ms=10)
plt.show()