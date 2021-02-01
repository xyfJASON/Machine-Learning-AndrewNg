"""
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=100, activation='relu', *, 
solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', 
learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
n_iter_no_change=10, max_fun=15000)
"""

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = loadmat('data/ex4data1.mat')
X = data['X']
m = X.shape[0]
X = np.transpose(X.reshape(m, 20, 20), [0, 2, 1]).reshape(m, 400)
y = data['y']
y[y==10] = 0
y = y.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

clf = MLPClassifier(hidden_layer_sizes=(25,), 
					random_state=True, 
					max_iter=1000)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
