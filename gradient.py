""" 
from gradient import cross_entropy_error , cross_entropy_error_label,
numerical_difference , numerical_gradient, gradient_descent

"""
import numpy as np

y1 = np.array([0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0])
t1 = np.array([0,0,0,1,0,0,0,0,0,0])

y2 = np.array([[0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0],
              [0.1, 0.05, 0, 0.06,0,0.1,0,0.4,0.5,0]])
t2 = np.array([[0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0]])
t1_label = np.array([3])
t2_label = np.array([3, 8])


def cross_entropy_error(y, t):
    """ 교차 엔트로피 에러값 구하기 원핫인코딩일 때의 예 """
    epsilon = 1e-7
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = t.shape[0]
    cee = -np.sum(t * np.log(y+ epsilon)) / batch_size
    return cee

# print(cross_entropy_error(y1, t1))
# print(cross_entropy_error(y2, t2))

def cross_entropy_error_label(y, t_label):
    """ 교차 엔트로피 에러값 구하기 레이블일 때 """
    epsilon = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t_label = t_label.reshape(1, t_label.size)
        
    batch_size = y.shape[0]
    cee = -np.sum(np.log(y[np.arange(batch_size), t_label])) / batch_size
    return cee

# print(cross_entropy_error_label(y1, t1_label))
# print(cross_entropy_error_label(y2, t2_label))
    

def f1(x):
    """ 미분 할 식 """
    return 0.01*x**2 + 0.1*x


def numerical_difference(f, x):
    """ 미분 """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
# print(numerical_difference(f1, 5))
# print(numerical_difference(f1, 10))

def f2(x):
    """ x제곱들의 합 """
    return np.sum(x**2)
# print(f2(t2_label))

def numerical_gradient(f, x):
    """ x is a vector containing input value of f at x"""
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

#     for i, x in enumerate(x): 
#         grad[i] = (f(x+h) - f(x-h)) / (2*h)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] =  tmp_val + h
        fxh1 = f(x)
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

x = np.array([-3.0, 4.0]) # Test Data X
# print(numerical_gradient(f2, x))

def gradient_descent(f, init_x, lr=0.1, epoch=100):
    x = init_x
    for i in range(epoch):
        x -= lr * numerical_gradient(f, x)
    return x

# print(gradient_descent(f2, x, 0.1, 100))
# print(gradient_descent(f2, x, 10.0,  100))
# print(gradient_descent(f2, x, 1e-4, 100000))

