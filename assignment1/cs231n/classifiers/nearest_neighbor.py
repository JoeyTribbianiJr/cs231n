'''

file: nearest_neighbor.py
project: classifiers
author: Joey Tribbiani (joeyfrancistribbiani@outlook.com)
file created: Wednesday, 20th December 2015 8:23:09 pm
-----
last modified: Wednesday, 20th December 2015 8:23:09 pm
modified by: Joey Tribbiani (joeyfrancistribbiani@outlook.com>)
-----
Copyright © 2014-2015 Phoebe Buffay, Xing Xin Internet cafes.

'''
import os
# sys.path.append('..')
import numpy as np
from ..data_utils import *
# from data_utils import *


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distance)
            ypred[i] = self.ytr[min_index]
            print('????: %.2f%%' % i / num_test)

        return ypred


def test():
    Xtr, Ytr, Xte, Yte = load_CIFAR10(os.getcwd() +
                                      '/cs231n/datasets/cifar-10-batches-py/')
    # Xtr_rows becomes 50000 x 3072
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    # Xte_rows becomes 10000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)

    yte_pred = nn.predict(Xte_rows)
    print('accuracy: %f' % (np.mean(yte_pred == Yte)))
