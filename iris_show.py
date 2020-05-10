# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from knn import KNN

iris = load_iris()
# print(iris)

X = iris.data#[:, :2]  # iris data input

# y의 label을 두개로 하고싶다면
# y = np.where(iris.target==2, 0, iris.target) 
y = iris.target # iris target (label)

y_name = iris.target_names # label 2개로 할 땐 [:2] 붙여주자
# print(y_name)

# Use 14/15 for training & 1/14 for testing
l = 15
for_test = np.array([i%l == (l-1) for i in range(y.shape[0])])
for_train = ~for_test

# for training data (X, y)
X_train = X[for_train]
y_train = y[for_train]

# for testing data (X, y)
X_test = X[for_test]
y_test = y[for_test]

def KNN_Classification(K, X_train, y_train, y_name):
    knn_iris = KNN(K, X_train, y_train, y_name)
    knn_iris.show_dim()
    print("k = ", K, ", majority_vote")
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])
        print("Test Data: ", i, " Computed class: ", knn_iris.majority_vote(),
             ",\tTrue class: ", y_name[y_test[i]])
        knn_iris.reset()

    print("k = ", K, ", weighted_majority_vote")
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])
        print("Test Data: ", i, " Computed class: ", knn_iris.weighted_majority_vote(),
             ",\tTrue class: ", y_name[y_test[i]])
        knn_iris.reset()
    print("")
KNN_Classification(3, X_train, y_train, y_name)    # K=3일 때
KNN_Classification(5, X_train, y_train, y_name)    # K=5일 때
KNN_Classification(10, X_train, y_train, y_name)   # K=10일 때

''' Cost Function Plotting '''
MVError = []    #majority_vote 성공갯수 x/10
WMVError = []   #weighted_majority_vote 성공갯수 x/10
def Cost(K, X_train, y_train, y_name):
    knn_iris = KNN(K, X_train, y_train, y_name)
    err=0
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])
        CC = knn_iris.majority_vote()
        TC = y_name[y_test[i]]
        if CC == TC:
            err += 1 
        knn_iris.reset()
    MVError.append(err)
    
    err=0
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])
        CC = knn_iris.weighted_majority_vote()
        TC = y_name[y_test[i]]
        if CC == TC:
            err += 1 
        knn_iris.reset()
    WMVError.append(err)
    
for K in range(1, 17):
    Cost(K, X_train, y_train, y_name)
X = np.arange(1, 17)
plt.figure(2, figsize=(8, 6))
plt.ylim(4, 10.5)
plt.plot(X,MVError,'bs:', X, WMVError, 'g^-')
plt.xlabel('K')
plt.ylabel('Cost')
plt.title(" Cost Function ")
plt.legend(["Majority Vote","Weighted Majority Vote"])
plt.show()