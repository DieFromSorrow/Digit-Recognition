#coding:utf-8
import numpy as np
from network.net import ConvNeuralNet
from network.optimizers import Optimizer
import pickle
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, 
        convnet: ConvNeuralNet, 
        optimizer: Optimizer, 
        x_train: np.ndarray, 
        t_train: np.ndarray,
        x_test: np.ndarray,
        t_test: np.ndarray
    ) -> None:
        self.convnet = convnet
        self.optimizer = optimizer
        self.x_train: np.ndarray = x_train
        self.t_train: np.ndarray = t_train
        self.x_test: np.ndarray = x_test
        self.t_test: np.ndarray = t_test
        self.train_size: int = self.x_train.shape[0]
        self.test_size: int = self.x_test.shape[0]

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        print('\n===  ===  ===  ===  ===  ===  ===  ===\n')
        print('        Initializated Trainer!')
        print('\n===  ===  ===  ===  ===  ===  ===  ===\n\n')

    def train(self, batch_size: int, epochs: int, test_num: int = 2000) -> None:
        for i in range(1, epochs + 1):
            train_batch_mask = np.random.choice(self.test_size, test_num)
            test_batch_mask = np.random.choice(self.test_size, test_num)
            print('---- Calculating Accuracy... ----')
            train_acc = self.convnet.accurate(x=self.x_train[train_batch_mask], t=self.t_train[train_batch_mask], batch_size=100)
            test_acc = self.convnet.accurate(x=self.x_test[test_batch_mask], t=self.t_test[test_batch_mask], batch_size=100)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            print('\n========= epoch:', i, '/', epochs, 'begin,', 'train_acc:', train_acc, 'test_acc:', test_acc, '=========\n')
            for _ in range(int(self.train_size / batch_size)):
                self.train_step(batch_size=batch_size)
        final_acc = self.convnet.accurate(self.x_test, self.t_test, batch_size=100)
        print('\n=================== Train End ===================\n')
        print('Final Test Accurate:', final_acc)
        print('\n')

    def train_step(self, batch_size: int) -> None:
        batch_mask = np.random.choice(self.train_size, batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        loss = self.convnet.gradient(x_batch, t_batch)
        self.optimizer.update(*self.convnet.forUpdate())
        self.train_loss_list.append(loss)
        print('training >>> train_loss:', loss)
        
    def save_trained_convnet(self, file_name: str) -> None:
        with open(file=file_name, mode='wb') as file_obj:
            pickle.dump(obj=self.convnet, file=file_obj)
        print('Saved Net Object Successfully In File: ' + file_name)

    def draw(self, file_name: str):
        plt.subplot(2,1,1)
        plt.xlabel('epochs')
        plt.ylabel('accurate')
        plt.plot(np.arange(len(self.train_acc_list)), np.array(self.train_acc_list), label='train')
        plt.plot(np.arange(len(self.test_acc_list)), np.array(self.test_acc_list), label='test')
        plt.legend()
        plt.subplot(2,1,2)
        plt.xlabel('batchs')
        plt.ylabel('loss')
        plt.plot(np.arange(len(self.train_loss_list)), np.array(self.train_loss_list))
        plt.savefig(file_name)
        print("Saved Training Image Successfully In File: " + file_name)
        plt.show()

