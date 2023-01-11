#coding:utf-8
import sys, os
sys.path.append(os.pardir)

from network import *
from dataset import load_mnist


if __name__ == "__main__":

    batch_size = 100    # 批大小
    epochs = 10         # 完整的数据集通过了神经网络一次并且返回了一次为一个 epoches
    test_num = 2000     # 测试批大小

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    convnet = ConvNeuralNet()
    optimizer = Adam()
    trainer = Trainer(
        convnet=convnet,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test
    )
    path = '../others/'
    trainer.train(batch_size=batch_size, epochs=epochs, test_num=500)
    trainer.save_trained_convnet(path + 'trained_net.pkl')
    trainer.draw(path + 'acc_loss_plot.jpg')
