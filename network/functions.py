#coding:utf-8
'''
functions for ConvNet
contains:
    ReLU
    SoftMax
    Cross-Entropy-Error
'''

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    '''
    x: 张量，任何维度数据
    y = x > 0 ? x : 0
    '''
    return x * (x > 0)


def softmax(x):
    '''
    x: 向量及矩阵
    y = exp(x[i]) / sigma(x[i], i: 1->n])
    '''
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    '''
    y: 向量或矩阵
    t: 监督数据，one-hot-vector模式或者普通模式
    y = -sigma(t[k] * ln(y[k]), k: 1->n)
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
