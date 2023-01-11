#coding:utf-8
import numpy as np
from network.layers import *
from collections import OrderedDict


'''
conv - relu - conv - relu - pool * 3
flatten
affine - relu - batchnorm - dropout * 2
affine - relu - affine - softmax_loss
'''
class ConvNeuralNet:
    def __init__(self,
        input_dim: tuple=(1, 28, 28),       # 输入形状
        weight_init_type='he',              # 权值初始化方式：'he' / 'xavier'
        layers_dict: OrderedDict={          # 层字典
            'conv1': {'filter_num': 16, 'filter_channels': 1, 'filter_height': 3, 'filter_width': 3, 'pad': 1, 'stride': 1},
            'relu1': {},
            'conv2': {'filter_num': 16, 'filter_channels': 16, 'filter_height': 3, 'filter_width': 3, 'pad': 1, 'stride': 1},
            'relu2': {},
            'pool1': {'pool_height': 2, 'pool_width': 2, 'pad': 0, 'stride': 2},
            'conv3': {'filter_num': 32, 'filter_channels': 16, 'filter_height': 3, 'filter_width': 3, 'pad': 1, 'stride': 1},
            'relu3': {},
            'conv4': {'filter_num': 32, 'filter_channels': 32, 'filter_height': 3, 'filter_width': 3, 'pad': 2, 'stride': 1},
            'relu4': {},
            'pool2': {'pool_height': 2, 'pool_width': 2, 'pad': 0, 'stride': 2},
            'conv5': {'filter_num': 64, 'filter_channels': 32, 'filter_height': 3, 'filter_width': 3, 'pad': 1, 'stride': 1},
            'relu5': {},
            'conv6': {'filter_num': 64, 'filter_channels': 64, 'filter_height': 3, 'filter_width': 3, 'pad': 1, 'stride': 1},
            'relu6': {},
            'pool3': {'pool_height': 2, 'pool_width': 2, 'pad': 0, 'stride': 2},
            'flatten1': {},                  # batch_size * 64 * 4 * 4 => batch_size * 1024
            'affine1': {'input_size': 1024, 'output_size': 392},
            'relu7': {},
            'batchnorm1': {},
            'dropout1': {'dropout_ratio': 0.5},
            'affine2': {'input_size':392, 'output_size': 196},
            'relu8': {},
            'batchnorm2': {},
            'dropout2': {'dropout_ratio': 0.2},
            'affine3': {'input_size':196, 'output_size': 49},
            'relu9': {},
            'batchnorm3': {},
            'affine4': {'input_size': 49, 'output_size': 10},
            'softmax_loss1': {}
        }
    ) -> None:
        
        self.weight_init_type = weight_init_type
        self.pre_layer_node_num = input_dim[0] * input_dim[1] * input_dim[2]
        
        self.layer_list = list()
        for key, val in layers_dict.items():
            self.layer_list.append(self.__initLayer(layer_key=key, layer_params=val))

    def __initLayer(self, layer_key: str, layer_params: dict) -> Layer :
        layer = None
        weight_init_tmpval = None

        if self.weight_init_type == 'he':
            weight_init_tmpval = 2
        elif self.weight_init_type == 'xavier':
            weight_init_tmpval = 1
            
        if layer_key[:4] == 'conv':
            weight_init_scale = weight_init_tmpval / (self.pre_layer_node_num ** (1/2))
            layer = self.__initConv(**layer_params, weight_init_scale=weight_init_scale)
            self.pre_layer_node_num = layer_params['filter_channels'] * layer_params['filter_height'] * layer_params['filter_width']
        elif layer_key[:4] == 'pool':
            layer = self.__initPool(**layer_params)
        elif layer_key[:4] == 'relu':
            layer = self.__initRelu(**layer_params)
        elif layer_key[:6] == 'affine':
            weight_init_scale = weight_init_tmpval / (self.pre_layer_node_num ** (1/2))
            layer = self.__initAffine(**layer_params, weight_init_scale=weight_init_scale)
            self.pre_layer_node_num = layer_params['output_size']
        elif layer_key[:7] == 'flatten':
            layer = self.__initFlatten(**layer_params)
        elif layer_key[:7] == 'dropout':
            layer = self.__initDropout(**layer_params)
        elif layer_key[:9] == 'batchnorm':
            layer = self.__initBatchNorm(**layer_params, input_size=self.pre_layer_node_num)
        elif layer_key[:12] == 'softmax_loss':
            layer = self.__initSoftmaxWithLoss(**layer_params)
        
        return layer

    def __initConv(self, filter_num: int, filter_channels: int, filter_height: int, filter_width: int, pad: int, stride: int, weight_init_scale: float) -> Convolution:
        filter_W = weight_init_scale * np.random.randn(filter_num, filter_channels, filter_height, filter_width)
        bias = np.zeros(filter_num)
        conv = Convolution(W=filter_W, b=bias, pad=pad, stride=stride)
        return conv
    
    def __initPool(self, pool_height: int, pool_width: int, pad: int, stride: int) -> Pooling:
        pool = Pooling(pool_h=pool_height, pool_w=pool_width, pad=pad, stride= stride)
        return pool

    def __initRelu(self) -> ReLU:
        relu = ReLU()
        return relu
    
    def __initFlatten(self) -> Flatten:
        flatten = Flatten()
        return flatten

    def __initAffine(self, input_size: int, output_size: int, weight_init_scale: float) -> Affine:
        Weights = weight_init_scale * np.random.randn(input_size, output_size)
        bias = np.zeros(output_size)
        affine = Affine(W=Weights, b=bias)
        return affine

    def __initBatchNorm(self, input_size):
        gamma = np.ones(input_size)
        beta = np.zeros(input_size)
        batch_norm = BatchNormalization(gamma=gamma, beta=beta)
        return batch_norm
    
    def __initDropout(self, dropout_ratio):
        dropout = Dropout(dropout_ratio=dropout_ratio)
        return dropout
    
    def __initSoftmaxWithLoss(self):
        softmax_loss = Softmax_Loss()
        return softmax_loss

    def predict(self, x: np.ndarray, train_flg: bool) -> np.ndarray:
        # print(x[0])
        for layer in self.layer_list[:-1]:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer.forward(x=x, train_flg=train_flg)
            else:
                x = layer.forward(x)
        # print(x[0])
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        y = self.predict(x=x, train_flg=True)
        loss = self.layer_list[-1].forward(y, t)
        return loss

    def accurate(self, x: np.ndarray, t: np.ndarray, batch_size=100) -> float:
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        correct = 0
        for i in range(int(x.shape[0] / batch_size)):
            x_batch = x[i * batch_size : (i + 1) * batch_size]
            t_batch = t[i * batch_size : (i + 1) * batch_size]
            y_batch = self.predict(x=x_batch, train_flg=False)
            y_batch = np.argmax(y_batch, axis=1)
            correct += np.sum(y_batch == t_batch)

        return correct / x.shape[0]

    def gradient(self, x: np.ndarray, t: np.ndarray) -> float:
        # print(x.shape, t.shape)
        loss = self.loss(x=x, t=t)
        self.layer_list.reverse()
        dout = 1.0
        for layer in self.layer_list:
            dout = layer.backward(dout)
        self.layer_list.reverse()
        return loss

    def forUpdate(self) -> list:
        params = {}
        grads = {}
        for layer in self.layer_list:
            if hasattr(layer, 'forUpdate'):
                p_g = layer.forUpdate()
                for key, val in p_g['params'].items():
                    params[key] = val
                for key, val in p_g['grads'].items():
                    grads[key] = val
        return (params, grads)