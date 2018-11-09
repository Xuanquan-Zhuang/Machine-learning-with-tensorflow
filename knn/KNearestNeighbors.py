#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Messiah
# @Date  : 18-9-18

import numpy as np
import tensorflow as tf
<<<<<<< HEAD:KNN/k_nearest_neighbors.py
from Datasets.DatasetsLoader import DatasetsLoader
=======
from sklearn.neighbors import KNeighborsClassifier
from datasets.DatasetsLoader import DatasetsLoader
>>>>>>> f258f97db77a74e4bf576a34b155ce402a9b6633:knn/KNearestNeighbors.py


class KNearestNeighbors:

    def __init__(self, neighbors, distance='l2'):
        self.neighbors = neighbors
        self.distance = distance
        self.features = None
        self.labels = None

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.float32, shape=[None, None])
            self.input_y = tf.placeholder(tf.int16, shape=[None, ])
            self.test_x = tf.placeholder(tf.float32, shape=[1, None])

            self.difference = tf.subtract(self.input_x, self.test_x)
            self.distance_l1 = tf.reduce_sum(tf.abs(self.difference), reduction_indices=1)
            self.distance_l2 = tf.reduce_sum(tf.pow(self.difference, 2), reduction_indices=1)

            _, self.top_k_index_l1 = tf.nn.top_k(tf.reshape(tf.negative(self.distance_l1), [1, -1]), self.neighbors)
            _, self.top_k_index_l2 = tf.nn.top_k(tf.reshape(tf.negative(self.distance_l2), [1, -1]), self.neighbors)

            self.target_l1, _, self.count_l1 = tf.unique_with_counts(tf.gather(self.input_y, self.top_k_index_l1[0]))
            self.target_l2, _, self.count_l2 = tf.unique_with_counts(tf.gather(self.input_y, self.top_k_index_l2)[0])

            self.output_l1 = self.target_l1[tf.argmax(self.count_l1, 0)]
            self.output_l2 = self.target_l2[tf.argmax(self.count_l2, 0)]

        self.sess = tf.Session(graph=self.graph)

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise Exception('length of X and y do not match')
        self.features = X
        self.labels = y

    def predict(self, X):
        prediction = []

        if self.distance == 'l2':
            for sample in X:
                prediction.append(self.sess.run(self.output_l2,
                                                feed_dict={self.input_x:self.features,
                                                           self.input_y:self.labels,
                                                           self.test_x:sample.reshape(1, -1)}))
        elif self.distance == 'l1':
            for sample in X:
                prediction.append(self.sess.run(self.output_l1,
                                                feed_dict={self.input_x:self.features,
                                                           self.input_y:self.labels,
                                                           self.test_x:sample.reshape(1, -1)}))

        return np.array(prediction)


if __name__ == '__main__':
    NEIGHBORS = 30
    DISTANCE = 'l2'

    data_loader = DatasetsLoader()
    fashion_mnist = data_loader.fashion_mnist_loader()

    x_train = fashion_mnist.train.images/255
    y_train = fashion_mnist.train.labels

    x_test = fashion_mnist.test.images/255
    y_test = fashion_mnist.test.labels

    model = KNearestNeighbors(NEIGHBORS, DISTANCE)

    model.fit(x_train, y_train)
    pred = model.predict(x_test[:50])

    acc = np.mean(np.equal(pred, y_test[:50]))

    print(pred)
    print(y_test[:50])
    print('Model: K Nearest Neighbors Classifier.')
    print('Neighbors: ' + str(NEIGHBORS) + ', distance: ' + DISTANCE)
    print('Accuracy: ' + str(acc))
