import numpy as np

class Logistic_Regression():
    def __init__(self, X_train, y_train, learningRate, S):
        self.X_train = self.Xmake(X_train) # x0 붙여서
        self.y_train = self.one_hot(y_train) # 원핫인코딩, 0과 1을 바꾼 matrix를 넘긴다.
        self.W = np.random.randn(X_train.shape[1] + 1, np.unique(y_train).shape[0]) # 초기 Weight값 ( # of feature by # of target )
        self.learningRate = learningRate
        self.h = []
        self.cost2=[]
        self.EMIN = -np.log(np.finfo(type(0.1)).max)
        self.Single = S
    
    # X_train 데이터에 bias 추가
    def Xmake(self, X_train):
        # z = [ [1], ... , [1] ] m개 , shape = (m, 1)
        z = np.ones(X_train.shape[0]).reshape(-1,1) 
        return np.concatenate((z, X_train), axis=1)
    
    def one_hot(self, y_train):
        # 이미 원 핫 인코딩이 된 경우(2차원)
        if y_train.ndim == 2:
            return y_train
        
        # 1차원 ndarray인 경우
        return np.eye(np.unique(y_train).shape[0])[y_train]
    
    def sigmoid(self, x):
        xSafe = np.array(np.maximum(x, self.EMIN))
        return(1.0/(1+np.exp(-xSafe)))

    def cost(self):
        # CostFunction, J()값의 추이를 볼 것
        vec1 = -(np.log(self.h) * self.y_train)
        J1 = vec1.sum(0) / self.y_train.sum(0)
        vec2 = -(np.log(np.maximum(1 - self.h, 1e-8)) * (1-self.y_train)) # log(0)이 안되므로 1e-8
        J2 = vec2.sum(0) / self.y_train.sum(0)
        # 각 Class 는 해당 Class의 logh(x) + 다른 Class의 1-logh(x)
        J = J1 + J2
        if self.Single == True:
            J = J[0]
        self.cost2.append(J) # plt위해 저장
        return J
        
    def learn(self, epoch):
        for i in range(epoch):
            # sigmoid 적용 h(w*x) => m*t matrix
            self.h = self.sigmoid(np.dot(self.X_train, self.W))
            self.W -= self.learningRate * np.dot(self.X_train.T,(self.h - self.y_train))
            print("epoch: ", i, "\t cost: ", self.cost())
        
    def predict(self, X_test, y_test):
        # MultiClass
        if self.Single == False:
            X = self.Xmake(X_test)
            # X_test값에 weight 적용
            Xweight = np.dot(X, self.W)
            # 각 데이터의 최대값 class 번호
            y_predict = np.argmax(Xweight, axis=1)
            # Accuracy %
            print("Accuracy = ",np.sum((y_test - y_predict)==0) / y_test.shape[0] * 100, "%")
        
        else :
            X = self.Xmake(X_test)
            # X_test값에 weight 적용
            Xweight = np.dot(X, self.W)
            # Xweight에 sigmoid를 거쳐 나온 값이 0.5이상인 것만 추출
            y_predict = np.array(self.sigmoid([Xweight])>=0.5)
            y_acc = np.equal(y_predict.reshape(-1, 2), self.one_hot(y_test))
            # Accuracy %
            print("Accuracy = ",y_acc.sum(0)[0] / y_test.shape[0] * 100,"%")
            
    def flush(self):
        self.W = []
        self.h = []
