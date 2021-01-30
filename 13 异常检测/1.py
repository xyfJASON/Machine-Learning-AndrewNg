import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = loadmat('ex8data1.mat')
X = data['X']
Xtmp = data['Xval']
ytmp = data['yval'].flatten()
Xval, Xtest, yval, ytest = train_test_split(Xtmp[ytmp==0], ytmp[ytmp==0], train_size=0.5)
Xval1, Xtest1, yval1, ytest1 = train_test_split(Xtmp[ytmp==1], ytmp[ytmp==1], train_size=0.5)
Xval, Xtest, yval, ytest = np.concatenate((Xval, Xval1)), np.concatenate((Xtest, Xtest1)), \
	np.concatenate((yval, yval1)), np.concatenate((ytest, ytest1))

Xmeans = X.mean(axis=0)
Xcov = ((X - Xmeans).T @ (X - Xmeans)) / X.shape[0]
normDist = multivariate_normal(mean=Xmeans, cov=Xcov)

def calc(pred, real):
	tp = np.sum(np.logical_and(pred==1, real==1))
	fp = np.sum(np.logical_and(pred==1, real==0))
	fn = np.sum(np.logical_and(pred==0, real==1))
	prec = 0 if tp + fp == 0 else tp / (tp + fp)
	rec = 0 if tp + fn == 0 else tp / (tp + fn)
	f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
	return f1

besteps, bestf1 = 0, 0
pdf_X = normDist.pdf(X)
pdf_Xval = normDist.pdf(Xval)
pdf_Xtest = normDist.pdf(Xtest)
for eps in np.linspace(pdf_Xval.min(), pdf_Xval.max(), 10000):
	f1 = calc(pdf_Xval < eps, yval)
	if f1 > bestf1:
		bestf1, besteps = f1, eps

print('besteps is:', besteps)
print('f1 on test set:', calc(pdf_Xtest < besteps, ytest))

x1, x2 = np.meshgrid(np.linspace(0, 30, 100), np.linspace(0, 30, 100))
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Throughput (mb/s)')
ax.plot(X[:, 0], X[:, 1], 'x', color='blue', alpha=0.5)
ax.plot(X[pdf_X<besteps][:, 0], X[pdf_X<besteps][:, 1], 'o', color='red', ms=10, markerfacecolor='none')
plt.show()
