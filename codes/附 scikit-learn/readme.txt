LinearRegression.py: 
	一元函数的线性回归
	使用 linear_model.LinearRegression

LinearRegression2.py: 
	多元函数的线性回归
	使用 linear_model.LinearRegression

LogisticRegression.py: 
	线性逻辑回归
	使用 linear_model.LogisticRegression

LogisticRegression2.py: 
	非线性逻辑回归
	首先使用 preprocessing.PolynomialFeatures 将特征扩展为多项式，
	然后使用 linear_model.LogisticRegression ，需要手动选择正则化系数 C；
	或者使用 linear_model.LogisticRegressionCV ，通过交叉验证自动调整正则化系数

PolynomialRegression.py:
	多项式回归
	首先使用 preprocessing.PolynomialFeatures 将特征扩展为多项式，
	然后使用 linear_model.Ridge 进行岭回归，即加入了 L2 正则化的线性回归，alpha 为正则化系数；
	或者使用 linear_model.RidgeCV，通过交叉验证自动调整正则化系数

BPNN.py:
	BP 神经网络
	首先使用 model_selection.train_test_split() 函数将数据集划分为训练集和测试集
	然后使用 neural_network.MLPClassifier 进行神经网络的训练

LinearSVC.py:
	线性可分支持向量机
	使用 svm.LinearSVC 进行分类

SVC.py:
	带核函数的支持向量机
	使用 svm.SVC 进行分类

KMeans.py:
	K-Means 算法
	使用 cluster.KMeans

PCA.py:
	主成分分析
	使用 decomposition.PCA，指定压缩的维度

PCA2.py:
	主成分分析
	使用 decomposition.PCA，选择维度使得 99% 的方差得以保留

AnomalyDetection.py:
	用高斯分布进行异常检测
	使用 covariance.EllipticEnvelope
	使用 metrics.f1_score 进行 F1 值的计算