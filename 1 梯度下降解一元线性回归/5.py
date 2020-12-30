import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
m = 0
eps = 1e-8
th0, th1 = 0, 0

with open("ex1data1.txt", "r") as infile:
	data = infile.readlines()
	for line in data:
		x, y = line.split(',')
		X.append(float(x))
		Y.append(float(y))
		m += 1

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

rep = 1000
alpha = 0.025
Z = []
for t in range(rep):
	nth0 = th0 - alpha * gradJ(0)
	nth1 = th1 - alpha * gradJ(1)
	th0 = nth0
	th1 = nth1
	Z.append(calc())
	if Z[-1] > 100: break;
plt1 = plt.subplot(131)
plt1.set_title("lr = 0.025")
plt.plot(range(1, len(Z)+1), Z, c = "green")

th0, th1 = 0, 0
alpha = 0.02
Z = []
for t in range(rep):
	nth0 = th0 - alpha * gradJ(0)
	nth1 = th1 - alpha * gradJ(1)
	th0 = nth0
	th1 = nth1
	Z.append(calc())
plt.subplot(132)
plt.title("lr = 0.02")
plt.plot(range(1, rep+1), Z, c = "blue")

th0, th1 = 0, 0
alpha = 0.01
Z = []
for t in range(rep):
	nth0 = th0 - alpha * gradJ(0)
	nth1 = th1 - alpha * gradJ(1)
	th0 = nth0
	th1 = nth1
	Z.append(calc())
plt.subplot(133)
plt.title("lr = 0.01")
plt.plot(range(1, rep+1), Z, c = "purple")

plt.show()