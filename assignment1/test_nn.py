'''

file: test_nn.py
project: assignment1
author: Joey Tribbiani (joeyfrancistribbiani@outlook.com)
file created: Wednesday, 20th December 2015 10:42:55 pm
-----
last modified: Wednesday, 20th December 2015 10:42:55 pm
modified by: Joey Tribbiani (joeyfrancistribbiani@outlook.com>)
-----
Copyright © 2014-2015 Phoebe Buffay, Xing Xin Internet cafes.

'''
import cs231n.classifiers.nearest_neighbor as nn_m
import cs231n.classifiers.k_nearest_neighbor as knn_m
import numpy as np
import pickle
from cs231n.data_utils import *


def nn_test():
    Xtr, Ytr, Xte, Yte = load_CIFAR10(os.getcwd() +
                                      '/cs231n/datasets/cifar-10-batches-py/')
    # Xtr_rows becomes 50000 x 3072
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    # Xte_rows becomes 10000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    nn = nn_m.NearestNeighbor()
    nn.train(Xtr_rows, Ytr)

    yte_pred = nn.predict(Xte_rows)
    print('accuracy: %f' % (np.mean(yte_pred == Yte)))


def knn_test():
    Xtr, Ytr, Xte, Yte = load_CIFAR10(os.getcwd() +
                                      '/cs231n/datasets/cifar-10-batches-py/')
    # Xtr_rows becomes 50000 x 3072
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    # Xte_rows becomes 10000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    Xval_rows = Xtr_rows[:100, :]  # take first 1000 for validation

    Yval = Ytr[:100]
    Xtr_rows = Xtr_rows[100:300, :]  # keep last 49,000 for train
    Ytr = Ytr[100:300]

    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20]:

        # use a particular value of k and evaluation on validation data
        nn = knn_m.KNearestNeighbor()
        nn.train(Xtr_rows, Ytr)
        # here we assume a modified NearestNeighbor class that can take a k as input
        Yval_predict = nn.predict(Xval_rows, k=k)
        acc = np.mean(Yval_predict == Yval)
        print('accuracy: %f' % (acc,))

        # keep track of what works on the validation set
        validation_accuracies.append((k, acc))

    print(validation_accuracies)
    pickle.dump(validation_accuracies, 'vali_acc.json')


if __name__ == "__main__":
    knn_test()
