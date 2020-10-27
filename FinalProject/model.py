# coding: utf-8
# 2020/인공지능/final/B511074/박준형
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


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
#         eMIN = -np.log(np.finfo(type(0.1)).max)
#         x = np.array(np.maximum(x, eMIN))
        self.out = 1.0 / (1 + np.exp(-x))
        
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        
        return dx

class tanh:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = np.tanh(x)
        
        return self.out
    
    def backward(self, dout):
        dx = dout * (1 - self.out**2)
        
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
            # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
                self.v[key] = np.zeros_like(val)
                
            for key in params.keys():
                # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]
                
class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
                self.h[key] = np.zeros_like(val)
                
            for key in params.keys():
                # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)

class CustomOptimizer:
    """ Adam : Momentum 과 AdaGrad 를 융합한 방법 
    v, h 가 각각 최초 0으로 설정되어 학습 초반에 0으로 biased 되는 문제를 해결하기 위해 고안한 방법 
    """
    def __init__(self, lr = 0.0001):
        self.lr = lr                # learningRate
        self.B1 = 0.9               # 베타1 0~1사이 값
        self.B2 = 0.999             # 베타2 0~1사이 값
        self.t = 0                  # Initialize timestep
        self.epsilon = 1e-7         # 1e-8 or 1e-7 무관
        self.m, self.v = None, None
        
    def update(self, params, grads):
        # None 인경우 m, v 초기화
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                # dropOut를 위해 저장된 파라미터는 갱신하지 않으므로 Pass
                if key is 'mean' or key is 'std':
                    continue
                    
                #  Initialize 1st moment vector
                self.m[key] = np.zeros_like(val)
                #  Initialize 2nd moment vector
                self.v[key] = np.zeros_like(val)
        
        self.t += 1 # t = t + 1
        # 연산속도 높이기 위해 key와 관련없는 값 반복문에서 빼서 미리 계산, epsilon은 작은 값이라 영향X
        lr1 = self.lr * np.sqrt(1.0 - self.B2**self.t) / (1.0 - self.B1**self.t)
        
        for key in params.keys():
            # dropOut를 위해 저장된 파라미터는 무관하므로 Pass
            if key is 'mean' or key is 'std':
                continue
                
            # Update biased first moment estimate
            self.m[key] = (self.B1 * self.m[key]) + ((1 - self.B1) * grads[key])
            # Update biased second raw moment estimate
            self.v[key] = (self.B2 * self.v[key]) + (1 - self.B2) * grads[key]**2
            # Compute bias-corrected first moment estimate
            mt = self.m[key] # / (1.0 - self.B1**self.t) 미리 계산
            # Compute bias-corrected second raw moment estimate
            vt = self.v[key] # / (1.0 - self.B2**self.t) 미리 계산
            # Update parameters
            params[key] -= lr1 * mt / (np.sqrt(vt) + self.epsilon)

class Dropout:
    def __init__(self, dropout_ratio = 0.1):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg = False):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            
            return x * self.mask
        
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    
def weight_init_std(input, type = 'Relu'):
    """ 가중치 초깃값 """

    # he 초깃값 -> Relu
    if type is 'Relu':
        return np.sqrt(2.0 / input)
      
    # Xavier 초깃값 -> sigmoid, tanh
    elif type is 'Sigmoid' or type is 'tanh':
        return 1.0 / np.sqrt(input)
    
class Model:
    """
    네트워크 모델 입니다.

    """

    def __init__(self, mean = None, std = None, dropFlag = False, layer_unit = [6, 64, 64, 64, 64, 6], lr=0.0001):
        """
        클래스 초기화
        """
        
        self.dropFlag = dropFlag          # dropout 레이어 생성 여부
        self.params = {} 
        self.params['mean'] = mean        # 
        self.params['std'] = std
        
        self.W = {}                       # 초깃값 plot해보기 위해 설정했던 값
        self.layer_unit = layer_unit      # 레이어 [6, 64, 64, 64, 64, 6]
        self.layer_size = len(layer_unit) # 레이어 수
        self.__init_weight()              # 초기 파라미터 설정
        self.layers = OrderedDict()       # Ordered딕셔너리로 초기화
        self.last_layer = None            # softMaxwithLoss
        self.__init_layer()               # 레이어 설정
        self.optimizer = CustomOptimizer(lr) # Adam으로 optimizer설정
        
    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        
        # Input layer -> hidden layer -> hidden...
        for i in range(1, self.layer_size - 1):
            # Affine 레이어 초기화
            self.layers['Affine{}'.format(i)] = \
                Affine(self.params['W{}'.format(i)], self.params['b{}'.format(i)])
            
            # Activation Function으로 Sigmoid 사용
            self.layers['Sigmoid{}'.format(i)] = Sigmoid() 
            
            # Flag = True일 경우 dropout레이어 만든다.
            if self.dropFlag:
                self.layers['Dropout{}'.format(i)] = Dropout()
        
        # hidden layer -> output
        i = self.layer_size - 1
        self.layers['Affine{}'.format(i)] = \
            Affine(self.params['W{}'.format(i)], self.params['b{}'.format(i)])
        
        # 마지막 레이어는 SoftmaxWithLoss
        self.last_layer = SoftmaxWithLoss()
        
    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        
        # 파라미터 초기화 Model에 입력된 layer에 따라 만들어진다.
        for i in range(1, self.layer_size):
            # 초깃값 Sigmoid: 1 / np.sqrt(self.layer_unit[i - 1]) 곱해준다.
            self.params['W{}'.format(i)] = weight_init_std(self.layer_unit[i - 1], 'Sigmoid') \
                        * np.random.randn(self.layer_unit[i - 1], self.layer_unit[i]) 
            # i번째 레이어 unit수만큼 0으로 초기화
            self.params['b{}'.format(i)] = np.zeros(self.layer_unit[i])
            
            # 초기 Weight값 분포 확인(plot)위해 self.W에 저장
            self.W['W{}'.format(i)] = self.params['W{}'.format(i)].copy()
            self.W['b{}'.format(i)] = self.params['b{}'.format(i)].copy()
        
    def update(self, x, t, dropFlag = False):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        
        # Trainer의 Train_step에서 update의 경우에 dropFlag = True이므로 attribute로 저장해둔다.
        self.dropFlag = dropFlag
        
        # 각 계층 forward, backward 전파 후 결과 dw, db 받아온다.
        grads = self.gradient(x, t)
        # 파라미터 갱신 Adam 사용
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        
        # featureScaling
        x2 = x.copy()
        # dataload시 Model에서 받아온 mean, std를 이용한 Scaling
        x2 -= self.params['mean']
        x2 /= self.params['std']
        
        # 각 레이어 forward propagation 진행
        for key, layer in self.layers.items():
            # dropout 계층에서는 Train하는 경우에만 dropout을 적용하므로
            if "Dropout" in key:
                x2 = layer.forward(x2, self.dropFlag)
            # forward propagation 진행
            else:
                x2 = layer.forward(x2)
                
        # Trainer의 Update이외에는 dropout을 끄도록 해준다.
        self.dropFlag=False
        
        return x2

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
        # forward propagation
        self.loss(x, t)
        
        # backward propagation
        dout = self.last_layer.backward(1)
        
        # 모든 계층 리스트화
        la = list(self.layers.values())
        # 역전파를 위한 reverse
        la.reverse()
        
        # 모든 레이어 backPropagation 진행
        for layer in la:
            dout = layer.backward(dout)
        
        # 결과 저장
        grads = {}
        
        # backPropagation 후 각 Affine계층에 dw와 db를 grads 딕셔너리로 리턴
        for i in range(1, self.layer_size):
            grads['W{}'.format(i)] = self.layers['Affine{}'.format(i)].dw
            grads['b{}'.format(i)] = self.layers['Affine{}'.format(i)].db
                
        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {} # params['mean'], params['std'] 도 함께 dump한다.
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
            self.params[key] = val # params['mean'], params['std'] 도 함께 load된다.
        self.__init_layer()
