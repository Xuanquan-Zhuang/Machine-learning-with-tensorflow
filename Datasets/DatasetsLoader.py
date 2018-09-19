#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Messiah
# @Date  : 18-9-18

import os
import pickle
from tensorflow.examples.tutorials.mnist import input_data


class DatasetsLoader():

    def __init__(self):
        self.fashion_mnist = './fashion mnist'
        self.mnist = './mnist'
        self.cifar10 = './cifar10'

    def fashion_mnist_loader(self):
        return input_data.read_data_sets(self.fashion_mnist)

    def mnist_loader(self):
        return input_data.read_data_sets(self.mnist)

    def input_cifar10(self):
        x_train = []
        y_train = []
        for i in range(1, 6):
            x, y = self.__unpickle(os.path.join(self.cifar10, 'data_batch_' + str(i)))
            x_train = x_train + list(x)
            y_train = y_train + list(y)

    @staticmethod
    def __unpickle(file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
            x = dictionary[b'data']
            y = dictionary[b'labels']
        return x, y
