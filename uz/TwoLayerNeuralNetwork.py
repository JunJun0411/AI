#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import os
from sklearn.datasets import load_iris
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[101]:


class TwoLayerNeuralNetwork():
    '''
    a neural network with one hidden layer
    '''
    def __init__(self, input_size, hidden_size, output_size):
        '''
        initailize attributes W1, b1, W2, b2
        '''
        self.params = {}
        # W1: x --> hideen layer 
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        # W2: hidden layer --> output layer
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)
        
        self.input_size = input_size # 4
        self.output_size = output_size # 3
        self.hs = hidden_size 
        
        # (150 X 4) -> (4 X hidden unit) -> (hidden unit X 3)
        
    def init_data(self, X, t):
        '''
        학습 데이터 저장
        '''
        self.x = X
        self.t = t
     
    def sigmoid(self, z):
        '''
        activation function
        '''
        eMin = -np.log(np.finfo(type(0.1)).max)
        
        zSafe = np.array(np.maximum(z, eMin))
        return(1.0 / (1 + np.exp(-zSafe)))
    
    def softmax(self, x):
        '''
        softmax(identity function) for clasification
        softmax 적용해도 원소의 대소관계 변하지 않음 
        '''
        
        y = np.zeros_like(x)
        # X = 1차원일 경우 
        if x.ndim == 1:
            exp_a = np.exp(x - np.max(x))
            sum_exp_a = np.sum(exp_a)
            y = exp_a / sum_exp_a
        # X = 2차원일 경우
        else :
            for idx, x in enumerate(x):
                exp_a = np.exp(x - np.max(x))
                sum_exp_a = np.sum(exp_a)
                y[idx] = exp_a / sum_exp_a
        return y
    
    def cross_entropy(self, y, t):
        '''
        loss function , one-hot encoding
        '''
        #for batch 
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
    
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y)) / batch_size 
        
    def predict(self, x):
        '''
        given input x, calculate output using current params : W1, b1, W2, b2
        data 들어오면 learn을 통해 갱신 해놓은 param을 이용하여 label 분류
        '''
        # input --> hidden layer
        z2 = np.dot(x, self.params['W1']) + self.params['b1']
        a2 = self.sigmoid(z2)
        # hidden layer --> output
        z3 = np.dot(a2, self.params['W2']) + self.params['b2']
        #print(z3)
        y = self.softmax(z3)
        return y
    
    def loss(self, x, t):
        '''
        x : input data, t : real label
        손실함수의 값(cost)을 구함
        '''
        y = self.predict(x)
        loss = self.cross_entropy(y, t)
        return loss
    
    def accuracy(self, x, t):
        '''
        모델의 정확도 측정
        '''
        y = self.predict(x)
        y = np.argmax(y, axis = 1) # 예측 결과 가장 큰 확률은 갖는 index
        t = np.argmax(t, axis = 1) # one-hot encoding -> 1값을 갖는 index
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numer_grad(self, f, w):
        '''
        가중치가 W에 대한 loss function(=self.cross_entropy)의 기울기
         = 가중치가 W 일때 손실함수의 값이 어떻게 변하나 
        편미분한 행렬의 형상은 W와 같음
        '''
        # loss function의 각 W에서 편미분
        h = 1e-4 # 0.0001
        grad = np.zeros_like(w) #w와 형상이 같은 행렬
        
        # 1차원인 b가 인자로 들어왔을때
        if w.ndim == 1:
            for i in range(w.size):
                tmp = w[i]
                # f(x+h)
                w[i] = tmp + h
                xh1 = f(w)

                #f(x-h)
                w[i] = tmp - h
                xh2 = f(w)

                grad[i] = (xh1 - xh2) / (2*h)
                w[i] = tmp                 
        
        # 2차원 행렬인 W가 인자로 들어왔을때
        else: 
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    tmp = w[i][j]
                    # f(x+h)
                    w[i][j] = tmp + h
                    xh1 = f(w)

                    #f(x-h)
                    w[i][j] = tmp - h
                    xh2 = f(w)

                    grad[i][j] = (xh1 - xh2) / (2*h)
                    w[i][j] = tmp 

        return grad
        
    def numerical_gradient(self, x, t):
        '''
        각 layer의 가중치의 손실함수에 대한 기울기를 저장 
        '''
        f = lambda w : self.loss(x ,t)
        
        grad = {} # 기울기 저장
        # input -> hidden layer
        grad['W1'] = self.numer_grad(f, self.params['W1'])
        grad['b1'] = self.numer_grad(f, self.params['b1'])
        
        # hidden later -> output
        grad['W2'] = self.numer_grad(f, self.params['W2'])
        grad['b2'] = self.numer_grad(f, self.params['b2'])
        
        return grad
    
    def learn(self, batch, lr, epoch, file):
        '''
        학습데이터로 numerical gradient과정을 거쳐 cost값이 제일 작아지도록
        W를 갱신시킴 
        '''
        batch = min(batch, self.x.shape[0]) #self.x = train data
        cost_accrcy = np.zeros([epoch, 2]) #for plotting
        f = file
        
        for i in range(epoch):
            #print("epoch: ",i, "cost, accuracy: ", self.loss(self.x, self.t), self.accuracy(self.x, self.t))
            f.write("epoch: {} cost, accuracy: {}\t{}\n".format(i, self.loss(self.x, self.t), self.accuracy(self.x, self.t)))
            #plotting
            loss = self.loss(self.x, self.t)
            accrcy = self.accuracy(self.x, self.t)
            cost_accrcy[i] = loss, accrcy
              
            # 1 epoch = n iteration (n = 전체 데이터 크기 / 배치 크기)    
            for j in range(int(self.x.shape[0] / batch)):
                batch_mask = np.random.choice(self.x.shape[0], batch)
                x_batch = self.x[batch_mask]
                t_batch = self.t[batch_mask]

                #기울기 계산
                grad = self.numerical_gradient(x_batch, t_batch)

                #매개변수 갱신
                for key in ('W1', 'b1', 'W2', 'b2'):
                    self.params[key] -= lr * grad[key]
        
        #plotting 
        plt.plot(cost_accrcy)
        plt.xlabel(epoch)
        plt.legend(['cost', 'accrcy'])
        plt.title('batch_size ={} , hidden_size = {}, lr = {}, epoch = {}'.format(batch, self.hs, lr, epoch))
        #plt.show()
    
        plt.savefig('./twolayernn_pltimg/batch={},hidden={},lr={},ep={}.png'.format(batch, self.hs, lr, epoch))
        
        plt.clf()
    
                
        


# In[102]:


'''iris = load_iris()
X = iris.data #150X4
t = iris.target #150
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2)
X_train[0].shape

# one_hot encoding
num = np.unique(t_train, axis = 0)
num = num.shape[0]
t_train = np.eye(num)[t_train]

num2 = np.unique(t_test, axis = 0)
num2 = num2.shape[0]
t_test = np.eye(num2)[t_test]'''


# In[105]:


'''hidden_layer = [5, 10]
epoch = [1000, 5000]
learning_rate = [0.001, 0.01]
batch_size = [40, 60, 120]'''


# In[106]:


'''for bch in batch_size:
    for lr in learning_rate:
        for ep in epoch:
            for hlayer in hidden_layer:
                fname = 'batch_size ={} , hidden_size = {}, lr = {}, epoch = {}.txt'.format(bch, hlayer, lr, ep)
                f = open(fname, 'w', encoding = 'utf8')
                
                nn = TwoLayerNeuralNetwork(4, hlayer, 3)
                nn.init_data(X_train, t_train)
                nn.learn(batch = bch, lr = lr, epoch = ep, file = f)
                tr = nn.accuracy(X_train, t_train)
                te = nn.accuracy(X_test, t_test)
                print('Training Accuracy: ', tr)
                print('Test Accuracy: ', te)
                
                f.write('Training Accuracy: {}\n'.format(tr))
                f.write('Test Accuracy: {}'.format(te))
                f.close()'''


# In[ ]:




