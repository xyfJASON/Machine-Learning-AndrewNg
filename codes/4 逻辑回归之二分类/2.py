import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("ex2data1.txt", delimiter = ',')
(m, n) = data.shape

p1 = plt.subplot(111)
X0 = data[data[:, -1] == 0, :]
X1 = data[data[:, -1] == 1, :]
p1.set_xlabel("Exam 1 score")
p1.set_ylabel("Exam 2 score")
p1.scatter(X0[:, 0:1], X0[:, 1:2], marker = 'x', c = 'r')
p1.scatter(X1[:, 0:1], X1[:, 1:2], marker = 'o')

s1 = data[:,0].std(ddof=1)
m1 = data[:,0].mean()
s2 = data[:,1].std(ddof=1)
m2 = data[:,1].mean()
T = np.array([1.2677702,3.05550587,2.81891901])
plt.plot([0, (T[2]*m2/s2-T[0])/T[1]*s1+m1], \
		 [(T[1]*m1/s1-T[0])/T[2]*s2+m2, 0])
plt.xlim(30, 100)
plt.ylim(30, 100)
plt.show()