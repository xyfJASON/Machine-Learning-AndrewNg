import numpy
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex5data1.mat')
print(data)
X, y, Xval, yval, Xtest, ytest = \
data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']

fig, ax = plt.subplots(1, 3)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Train')
ax[0].plot(X, y, 'o', color='magenta')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Valid')
ax[1].plot(Xval, yval, 'o', color='dodgerblue')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_title('Test')
ax[2].plot(Xtest, ytest, 'o', color='forestgreen')
plt.show()