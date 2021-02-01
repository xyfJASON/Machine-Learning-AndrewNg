import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
m = 0
alpha = 0.01
eps = 1e-8
th0, th1 = 0, 0

def gradJ(k):
	res = 0
	for x, y in zip(X, Y):
		res += (th1 * x + th0 - y) * (x if k == 1 else 1)
	return res / m

def calc():
	res = 0
	for x, y in zip(X, Y):
		res += (th1 * x + th0 - y) * (th1 * x + th0 - y)
	return res / 2 / m

with open("ex1data1.txt", "r") as infile:
	data = infile.readlines()
	for line in data:
		x, y = line.split(',')
		X.append(float(x))
		Y.append(float(y))
		m += 1


while 1:
	nth0 = th0 - alpha * gradJ(0)
	nth1 = th1 - alpha * gradJ(1)
	if abs(th0 - nth0) < eps and abs(th1 - nth1) < eps:
		break
	th0 = nth0
	th1 = nth1
print(th0, th1)
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [th0+min(X)*th1, th0+max(X)*th1], c="magenta")
plt.show()