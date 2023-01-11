#coding:utf-8
'''
layers for ConvNet
contains:
    Flatten
    Affine
    ReLU
    Softmax_Loss
    BatchNomalization
    Dropout
    Convolution
    Pooling
'''

import numpy as np
from network.functions import *
from network.utils import *


class Layer:
    def __init__(self) -> None:
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.original_x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.original_x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(*self.original_x_shape)


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        super().__init__()
        self.W = W
        self.b = b
        self.x = None
        self.out = None
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.out = np.dot(self.x, self.W) + self.b
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        '''
        dx = dout * wT
        dw = xT * dout
        '''
        self.dout = dout
        self.dx = np.dot(self.dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return self.dx

    def forUpdate(self) -> None:
        return {
            'params': {
                'W' + str(id(self)): self.W,
                'b' + str(id(self)): self.b
            },
            'grads': {
                'W' + str(id(self)): self.dW,
                'b' + str(id(self)): self.db
            }
        }


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Softmax_Loss(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.loss = None
        self.y = None # softmax 输出
        self.t = None # 监督数据

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout: int=1) -> np.ndarray:
        '''
        dx = (y - t) * dout(1)
        '''
        batch_size = self.y.shape[0]
        if self.t.size != self.y.size:
            tmp = np.zeros_like(self.y)
            tmp[batch_size, self.t] = 1
            self.t = tmp
        dx = (self.y - self.t) / batch_size
        return dx


class BatchNormalization(Layer):
    def __init__(self,
        gamma: float, beta: float, momentum: float=0.9, running_mean: np.ndarray=None, running_var: np.ndarray=None
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 测试时用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgmma = None
        self.dbeta = None

    def forward(self, x: np.ndarray, train_flg:bool=True) -> np.ndarray:
        '''
        Input: {x1, x2, ..., xm}

        mean = (1/m) * sigma(xi, i: 1->m)
        var = (1/m) * sigma((x - mean) ** 2)
        xi <= (xi - mean) / sqrt(var + 1e-7)
        yi <= gamma * xi + beta

        output: {y1, y2, ..., ym}
        '''
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg=train_flg)
        return out.reshape(*self.input_shape)
    
    def __forward(self, x: np.ndarray, train_flg:bool) -> np.ndarray:
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        
        if train_flg:
            mean = x.mean(axis=0) # 均值
            xc = x - mean
            var = np.mean(xc**2, axis=0) # 方差
            std = np.sqrt(var + 1e-7)    # 标准差
            xn = xc / std                # 标准化后的 x

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))
        
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout: np.ndarray) -> np.ndarray:
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
    
    def forUpdate(self) -> dict:
        return {
            'params': {
                'gamma' + str(id(self)): self.gamma,
                'beta' + str(id(self)): self.beta
            },
            'grads': {
                'gamma' + str(id(self)): self.dgamma,
                'beta' + str(id(self)): self.beta
            }
        }


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.3) -> None:
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: np.ndarray, train_flg: bool=True) -> np.ndarray:
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Convolution(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray, stride: int=1, pad: int=0) -> None:
        super().__init__()
        self.W = W              # 滤波器权重
        self.b = b              # 偏置
        self.stride = stride    # 步长
        self.pad = pad          # 填充

        # 中间数据，方便 backward
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T    # 转置是为了方便 dot 运算

        out: np.ndarray = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW: np.ndarray = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
    def forUpdate(self):
        return {
            'params': {
                'W' + str(id(self)): self.W,
                'b' + str(id(self)): self.b
            },
            'grads': {
                'W' + str(id(self)): self.dW,
                'b' + str(id(self)): self.db
            }
        }


class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int=1, pad: int=0) -> None:
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx