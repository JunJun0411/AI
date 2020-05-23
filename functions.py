# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([-1, 1, 0, 800, -900, 10000])
x2 = np.array([-1, 4, 7, -9])

def step_function(x):
    return (np.array(x>0, dtype=np.int8))

# print(step_function(1))
# print(step_function(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.show()

def sigmoid(x):
    return (1 / (1+np.exp(-x)))

# print(sigmoid(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.show()

def reLU(x):
    return (np.maximum(x, 0))

# print(reLU(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = reLU(x)
# plt.plot(x,y)
# plt.show()

def softmax(x):
    exp_a = np.exp(x - np.max(x))
    sum_exp_a = np.sum(exp_a)
#     print(exp_a)
#     print(sum_exp_a)
    return exp_a / sum_exp_a

# print(Softmax(x1))
# print(np.sum(Softmax(x1)))

# x = np.arange(-5.0, 5.0, 0.1)
# y = reLU(x)
# plt.plot(x,y)
# plt.show()

