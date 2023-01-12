#coding:utf-8
import sys, os
sys.path.append(os.pardir)

from network import ConvNeuralNet
import numpy as np
import pickle
import random
from PIL import Image
from dataset import load_mnist


def test_network(filename: str, test_num: int) -> float:
    imgl = []

    def img_show(_img):
        pil_img = Image.fromarray(np.uint8(_img))
        pil_img.show()

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    (x_train, t_train), (imgs, tags) = load_mnist(normalize=False, flatten=False, one_hot_label=False)

    network: ConvNeuralNet = None

    with open(filename, 'rb') as f:
        network = pickle.load(f)
    
    pl = []
    tl = []
    right_num = 0
    
    for _ in range(test_num):

        idx = random.randint(0, len(x_test) - 1)

        x = np.array([x_test[idx]])
        t = np.array([t_test[idx]])

        img = imgs[idx]

        p = np.argmax(network.predict(x, train_flg=False))
        t = np.argmax(t)

        pl.append(p)
        tl.append(t)
        if p == t:
            right_num += 1
            
        imgl.append(img.reshape(28, 28))

    imgs = imgl[0]
    for i in range(1, len(imgl)):
        imgs = np.concatenate([imgs, imgl[i]], axis=1)

    print("forecast results:", pl)
    print("  actual results:", tl)

    img_show(imgs)

    return right_num / test_num
    

def acc(x, t, net: ConvNeuralNet):
    return net.accurate(x=x, t=t, batch_size=100)


if __name__ == "__main__":
    filename = '../others/trained_net.pkl'
    test_num = 25
    accurate = test_network(filename=filename, test_num=test_num)
    print(f"Prediction Accuracy: {accurate}")