# coding: utf-8

import numpy as np

y1 = np.array([0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0])
t1 = np.array([0,0,0,1,0,0,0,0,0,0])

y2 = np.array([[0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0],
              [0.1, 0.05, 0, 0.06,0,0.1,0,0.4,0.5,0]])
t2 = np.array([[0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0]])

# 교차 엔트로피 에러값 구하기
def cross_entropy_error(y, t):
    epsilon = 1e-7
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = t.shape[0]
    cee = -np.sum(t * np.log(y+ epsilon)) / batch_size
    return cee
print(cross_entropy_error(y1, t1))
print(cross_entropy_error(y2, t2))

# 미분 할 식
def f1(x):
    return 0.01*x**2 + 0.1*x

# 미분
def numerical_difference(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
    
print(numerical_difference(f1, 5))
print(numerical_difference(f1, 10))