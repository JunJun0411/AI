import numpy as np

x1 = np.array([[-1, 1, 0, 800, -900, 10000],
              [2, 4, 0, 1, 10, 5]])
x2 = np.array([-1, 4, 7, -9])
x3 = np.array([2, 4, 0, 1, 10, 5])

def step_function(x):
    """ 계단 함수 """
    return (np.array(x>0, dtype=np.int8))

# print(step_function(1))
# print(step_function(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.show()


def sigmoid(x):
    """ 시그모이드 함수 """
    EMIN = -np.log(np.finfo(type(0.1)).max)
    xSafe = np.array(np.maximum(x, EMIN))
    return(1.0/(1+np.exp(-xSafe)))

# print(sigmoid(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.show()

def reLU(x):
    """ reLU 함수 """
    return (np.maximum(x, 0))

# print(reLU(x1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = reLU(x)
# plt.plot(x,y)
# plt.show()

def softmax(x):
    """ softmax 다차원 가능 """
    if x.ndim == 1:
        x = x.reshape(1,-1)
        
    exp_a = np.exp(x - x.max(axis=1).reshape(-1,1))
    sum_exp_a = exp_a.sum(1).reshape(-1,1)
    return exp_a / sum_exp_a

# print(softmax(x1))
# print(softmax(x3))
# print(softmax(x3).sum(1))

# x = np.arange(-5.0, 5.0, 0.1)
# y = reLU(x)
# plt.plot(x,y)
# plt.show()