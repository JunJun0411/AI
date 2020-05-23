import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from Logistic_regression import Logistic_Regression

# X_train, y_train, X_test, y_test, epoch, learningRate, Single이라면 True, Snum: Single class number
def Logistic(X_train, y_train, X_test, y_test, epoch, learningRate, Single, Snum):
    #Single Class라면 해당 Class에 맞게 inputdata Slicing
    if Single == True:
        y_train = np.eye(np.unique(y_train).shape[0], dtype = int)[y_train]
        y_test = np.eye(np.unique(y_test).shape[0], dtype = int)[y_test]
        y_train = y_train[:, Snum]
        y_test = y_test[:, Snum]
    # Logistic Regression    
    lo = Logistic_Regression(X_train, y_train, learningRate, Single)
    lo.learn(epoch) # epoch만큼 학습
    if Single == True:
        print(Snum," Class, epoch = ", epoch, "LearningRate = ", learningRate)
    else:
        print("MultiClass, epoch = ", epoch, "LearningRate = ", learningRate)
    lo.predict(X_test, y_test)
    y = lo.cost2
    x = np.arange(epoch)
    plt.plot(x, y)
    plt.show()
    
#Mnist Load Data
(X_train, t_train), (X_test, t_test) = load_mnist(flatten=True, normalize=True) 

epoch = 100
learningRate = 0.00001
Single = False
ClassNum = -1 #Nulti의 경우
if Single == True: #Single일 경우 원하는 ClassNum로 변경
    ClassNum=0
    
# X_train, y_train, X_test, y_test, epoch, learningRate, Single이라면 True, Snum: Single class number
Logistic(X_train, t_train, X_test, t_test, epoch, learningRate, Single, ClassNum) # MultiClass로 진행
#Logistic(X_train, t_train, X_test, t_test, epoch, learningRate, Single, ClassNum) # SingleClass 0
