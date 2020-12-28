import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("ex2data2.txt", delimiter = ',')
(m, n) = data.shape

p1 = plt.subplot(111)
X0 = data[data[:, -1] == 0, :]
X1 = data[data[:, -1] == 1, :]
p1.set_xlabel("Microchip test 1")
p1.set_ylabel("Microchip test 2")
p1.scatter(X0[:, 0:1], X0[:, 1:2], marker = 'x', c = 'r')
p1.scatter(X1[:, 0:1], X1[:, 1:2], marker = 'o')
plt.show()