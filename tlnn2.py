import numpy as np
from functions import sigmoid, softmax

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
        self.input_size, self.output_size = input_size, output_size # input, output size 저장
        
    def predict(self, x):
        """calculate output given input and current parameters: W1, b1, W2, b2 """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # input --> hidden layer : sigmoid
        z2 = np.dot(x, W1) + b1
        a2 = sigmoid(z2)
        
        # hidden layer --> output : softmax
        z3 = np.dot(a2, W2) + b2
        y = softmax(z3)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
        
#     def accuracy(selfself, x, t):
#     def numerical_gradient(self, x, t):
#     def learn(self, lr, epoch):
