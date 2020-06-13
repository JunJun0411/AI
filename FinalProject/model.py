# coding: utf-8
# 2020/인공지능/final/B511074/박준형
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        
        return dout


class CustomActivation:
    """sigmoid"""
    def __init__(self):
        self.out = None

    def forward(self, x):
        eMIN = -np.log(np.finfo(type(0.1)).max)
        xSafe = np.array(np.maximum(x, eMIN))
        self.out = (1.0 / 1 + np.exp(-xSafe))
        
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x, self.dw, self.db = None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.t = None
        self.y = None

    def forward(self, x, t):
        if t.ndim == 1: #one hot 안되어 있는 경우
            t = np.eye(6)[t]
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class CustomOptimizer:
    pass

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > dropout_ratio
            
            return x * self.mask
        
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask

class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """

        self.params = {}
        self.__init_weight()
        self.layers = OrderedDict()
        self.last_layer = None
        self.__init_layer()
        self.optimizer = SGD(lr)

    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        
        # input -> 1 hidden layer
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        
        # 1 hidden layer -> 2 hidden layer
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
         
        self.last_layer = SoftmaxWithLoss()
        
        # 2 hidden layer -> 3 hidden layer
        

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        weight_init_std = 1
        input_size = 6
        hidden_size_1 = 10
        hidden_size_2 = 10
        output_size = 6
        
        self.params['W1'] = np.random.randn(input_size, hidden_size_1) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size_1)
        
        self.params['W2'] = np.random.randn(hidden_size_1, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)
    
    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        self.loss(x, t)
        
        # backward
        dout = self.last_layer.backward(1)
        
        la = list(self.layers.values())
        la.reverse()
        
        for layer in la:
            dout = layer.backward(dout)
        
        # 결과 저장
        grads = {}
        
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        pass