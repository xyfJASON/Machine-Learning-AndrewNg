import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

with open("ex1data1.txt", "r") as infile:
	data = infile.readlines()
	for line in data:
		x, y = line.split(',')
		X.append(float(x))
		Y.append(float(y))

plt.scatter(X, Y)
plt.show()