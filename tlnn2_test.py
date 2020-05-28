import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from tlnn2 import TwoLayerNeuralNetwork2

""" iris Data """
iris = load_iris()
X = iris.data # iris data input
y = iris.target # iris target (label)

# 데이터 Split Use training : testing = 8 : 2 => 120 : 30
suffle = np.random.choice(X.shape[0], X.shape[0], replace=False)
for_train = suffle[:120]
for_test = suffle[120:]

# for training data (X, y)
X_train = X[for_train]
y_train = y[for_train]
# for testing data (X, y)
X_test = X[for_test]
y_test = y[for_test]

""" hidden layer의 Unit 수 = 3, 5, 7"""
hidden_size = [3, 5, 7]
input_size = 4
output_size = 3

""" hyperParameter 값 변화하며 Test """
lr = [0.02, 0.01, 0.005] # learningRate 0.02, 0.01, 0.005
epoch = [10000] # epoch 1000, 5000, 10000
batch_size = [40, 60, 120] # batchSize 40, 60, 120
check = 1 # loss, accuracy print

for hid in hidden_size:
    tn2 = TwoLayerNeuralNetwork2(input_size, hid, output_size)
    tn2.init_data(X_train, y_train) # Set Training Data
    for e in epoch:
        for l in lr:
            for b in batch_size:
                lossPlt = []
                accPlt = []
                
                # 학습: epoch번 반복한 loss와 accuracy의 List를 return 
                lossPlt, accPlt = tn2.learn(l, e, b, check) 
                
                lo = round(tn2.loss(X_train, y_train), 5)
                Tr = round(tn2.accuracy(X_train, y_train), 5)
                Te = round(tn2.accuracy(X_test, y_test), 5)
                tn2.reset() # parameter값들 Reset 해주어야 한다.
                
                # loss와 training accuracy를 Plot
                x = np.arange(e)
                plt.plot(x, lossPlt, x, accPlt)
                plt.legend(["loss", "training accuracy"]) # 각주
                plt.title('hiddenSize: {}, lr: {}, epoch: {}, batchSize: {}\n loss: {}, Train_Acc: {}, Test_Acc: {}'                          .format(hid, l, e, b, lo, Tr, Te))
                plt.savefig('./Result/[hi_{}]_[lr_{}]_[ep_{}]_[ba_{}].png'.format(hid, l, e, b), dpi=100)
                plt.clf()

