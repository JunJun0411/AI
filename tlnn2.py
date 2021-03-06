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
        """ TrainingData를 class의 attribute로 지정"""
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
        """ x를 통해 들어온 데이터를 계산하여 도출한 y값과 실제 target값의 loss를 계산 """
        y = self.predict(x)
        
        # 만약 target으로 넘어온 값이 1차원 array인 label이라면
        if t.ndim==1:
            return cross_entropy_error_label(y, t)
        
        # target이 원 핫 인코딩되어 넘어왔다면
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

        # 새 딕셔너리 params에 현재 상태에 대한 Weight값과 bias값의 기울기를 저장한다.
        params = {}
        params['W1'] = numerical_gradient(f, self.params['W1'])
        params['b1'] = numerical_gradient(f, self.params['b1'])
        params['W2'] = numerical_gradient(f, self.params['W2'])
        params['b2'] = numerical_gradient(f, self.params['b2'])
        
        return params
        
    # check == 1이라면 loss, acc를 print, check == 2라면 file에 저장
    def learn(self, lr = 0.01, epoch = 100, batch_size = 1, check = 1):
        """ pre-requisite: x, t are stored in the local attribute"""
        # Plt 추이를 보기 위한 list 선언
        lossPlt, accPlt = [], []
        if check == 2:
            file = open('./Result/[hi_{}]_[lr_{}]_[ep_{}]_[ba_{}].txt'\
                        .format(self.hidden_size, lr, epoch, batch_size), 'w')
        # epoch 만큼 반복 수행
        for i in range(epoch):
            # 훈련데이터의 현재 loss값과 accuracy를 구한다.
            lo = self.loss(self.x, self.t)
            ac = self.accuracy(self.x, self.t)

            # 각 list에 추가한다. (plot을 보기 위함)
            lossPlt.append(lo)
            accPlt.append(ac)
            
            # loss값과 정확도를 확인하고 싶다면 check=1
            if check == 1:
                print(i, "번째 loss, accuracy: " , lo, ac)
            # loss값과 정확도를 file에 쓰고싶다면 check =2
            elif check == 2:
                file.write("%d번째 loss, accuracy: " % i)
                file.write("%f, " % lo)
                file.write("%f\n" % ac)
                file.close()
                
            # 0 ~ 훈련Data 중에 batch_size만큼 random으로 뽑아낸다.
            batch_size = min(batch_size, self.x.shape[0])
            # 학습 데이터 수 만큼 random choice(suffle)한다.(중복 X)
            suffle = np.random.choice(self.x.shape[0], self.x.shape[0], replace=False)

            # 전체 데이터 / batch_size 만큼 반복 한 것이 1 epoch이다. 
            for i in range(int(self.x.shape[0] / batch_size)):
                # x_train, t_train을 batch_size만큼 split한다.
                batch_mask = suffle[i * batch_size : (i + 1) * batch_size]
                x_train = self.x[batch_mask]
                t_train = self.t[batch_mask]

                # 미니 배치 훈련 데이터를 통해 기울기를 구한다.

                params = self.numerical_gradient(x_train, t_train)
                # 기울어진 방향으로 가중치(W1, b1, W2, b2)의 값을 갱신한다.
                for key in self.params:
                    self.params[key] -= lr * params[key]
        
        # loss와 accuracy의 Plt List들을 return
        return lossPlt, accPlt
        
    # Class의 변수들 초기화 -> lr, epoch, batchSize 변화 비교 실험 시 필요하다.
    def reset(self):
        """ reset parameters """
        self.params['W1'] = np.random.randn(self.input_size, self.hidden_size)
        self.params['b1'] = np.random.randn(self.hidden_size)
        self.params['W2'] = np.random.randn(self.hidden_size, self.output_size)
        self.params['b2'] = np.random.randn(self.output_size)