import numpy as np
import sys, os
sys.path.append(os.pardir)
from functions import sigmoid, softmax
from gradient import cross_entropy_error_label, cross_entropy_error, numerical_gradient

class TwoLayerNeuralNetwork2:
    """ a neural network with one hidden layer """
    def __init__(self, input_size, hidden_size, output_size):
        """ initialize parameters """
        self.params = {}
        # input --> hidden layer
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        # hidden layer --> output layer
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size # input, output size 저장
        self.x, self.t = [], []
        
    def init_data(self, x_train, t_train):
        self.x = x_train
        # 1차원 벡터라면 원 핫 인코딩을 해준다.
        if t_train.ndim == 1:
            t_train = np.eye(np.unique(t_train).shape[0])[t_train]
        self.t = t_train
        
        
    def predict(self, x):
        """calculate output given input and current parameters: W1, b1, W2, b2 """
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']
        
        # input --> hidden layer : sigmoid
        z2 = np.dot(x, W1) + b1
        a2 = sigmoid(z2)
        
        # hidden layer --> output : softmax
        z3 = np.dot(a2, W2) + b2
        y = softmax(z3)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        
        if t.ndim==1:
            return cross_entropy_error_label(y, t)
        return cross_entropy_error(y, t)
        
    def accuracy(self, x, t):
        """ testData로 실제 target과 계산된 y를 비교해서 정확도를 구한다. """
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        
        # target_y값이 원 핫 인코딩이 되어있다면 1차원의 ndarray로 변환한다.
        if t.ndim == 2:
            t = np.argmax(t, axis = 1)
        
        # 정확도는 target값과 계산된 y값이 같은 갯수를 test데이터 수로 나눈 확률 값
        accuracy = np.sum(y-t == 0) / t.shape[0]
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        """ 가중치 매개변수의 기울기를 구한다. """
        # lambda를 이용하여 loss(x, t)의 함수 f를 구한다.
        f = lambda W: self.loss(x, t)

        # 새 딕셔너리 params에 각 layer의 Weight값과 bias값의 기울기를 저장한다.
        params = {}
        params['W1'] = numerical_gradient(f, self.params['W1'])
        params['b1'] = numerical_gradient(f, self.params['b1'])
        params['W2'] = numerical_gradient(f, self.params['W2'])
        params['b2'] = numerical_gradient(f, self.params['b2'])
        
        return params
        
    # batch = True이면 batch 사용, check = True라면 값의 추이, plt를 확인한다.
    def learn(self, lr = 0.01, epoch = 100, batch_size = 1, batch = True, check = True):
        """ pre-requisite: x, t are stored in the local attribute"""
        # Plt 추이를 보기 위한 list 선언
        lossPlt, accPlt = [], []
        
        # epoch 만큼 반복 수행
        for i in range(epoch):
            # Plt를 보고 싶다면 check=True
            # lr,epoch,batchsize 변화 비교를 위해 불필요한 연산을 Skip하고 싶다면 check=False
            if check:
                # 훈련데이터의 현재 loss값과 accuracy를 구한다.
                lo = self.loss(self.x, self.t)
                ac = self.accuracy(self.x, self.t)
                
                # 각 list에 추가한다. (plot을 보기 위함)
                lossPlt.append(lo)
                accPlt.append(ac)
                print(i, "번째 loss, accuracy: " , lo, ac)
            
            # 훈련할 데이터 x값과 target을 x_train, t_train에 배정한다.
            x_train = self.x
            t_train = self.t
            
            # batch를 사용한다면
            if batch: 
                # 0 ~ 훈련Data 중에 batch_size만큼 random으로 뽑아낸다. 이때, 중복된 값도 허용된다.
                batch_mask = np.random.choice(self.x.shape[0], batch_size)
                x_train = self.x[batch_mask]
                t_train = self.t[batch_mask]
            
            # 훈련 데이터를 통해 기울기를 구한다.
            params = self.numerical_gradient(x_train, t_train)
            # 기울어진 방향으로 가중치의 값을 조정하고 learningRate를 곱함으로 overshooting을 막는다.
            for key in self.params:
                self.params[key] -= lr * params[key]
        
        # 만약 check=True라면 loss와 accuracy의 Plt List들을 return
        if check:
            return lossPlt, accPlt
        
    # Class의 변수들 초기화 -> lr, epoch, batchSize 변화 비교 실험 시 필요하다.
    def reset(self):
        """ reset parameters """
        self.params['W1'] = np.random.randn(self.input_size, self.hidden_size)
        self.params['b1'] = np.random.randn(self.hidden_size)
        self.params['W2'] = np.random.randn(self.hidden_size, self.output_size)
        self.params['b2'] = np.random.randn(self.output_size)