#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Messiah
# @Date  : 18-9-21

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston


class LinearRegression:

    def __init__(self, intercept=True, learning_rate=0.001):
        self.has_intercept = intercept
        self.learning_rate = learning_rate
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.float32, shape=[None, None])
            self.input_y = tf.placeholder(tf.float32, shape=[None, ])
            self.test_x = tf.placeholder(tf.float32, shape=[None, None])

            self.feature_num = tf.placeholder(tf.int16, shape=[1])

            self.initializer = tf.global_variables_initializer()
            self.coefficients = None
            self.intercept = None
            self.prediction = None
            self.loss = None
            self.train = None

        self.sess = tf.Session(graph=self.graph)

    def fit(self, x, y, batch_size=100, epochs=1000):
        with self.graph.as_default():
            self.coefficients = tf.Variable(tf.truncated_normal([x.shape[1]]), dtype=tf.float32)
            if self.has_intercept:
                self.intercept = tf.Variable(tf.truncated_normal([1]), dtype=tf.float32)
                self.prediction = tf.add(tf.matmul(self.input_x, self.coefficients), self.intercept)
            else:
                self.prediction = tf.matmul(self.input_x, self.coefficients)

            self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.prediction, self.input_y), 2),
                                       reduction_indices=0)
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.sess.run(self.initializer)

        size = len(x)
        mini_batch_size = batch_size if batch_size < size else size

        for epoch in range(epochs):
            np.random.shuffle(x)
            batch = 0
            while batch * mini_batch_size < size:
                if (batch + 1) * mini_batch_size < size:
                    self.sess.run(self.train, feed_dict={self.input_x: x[batch*mini_batch_size:(batch+1)*mini_batch_size],
                                                         self.input_y: y[batch*mini_batch_size:(batch+1)*mini_batch_size]})
                else:
                    self.sess.run(self.train, feed_dict={self.input_x: x[batch*mini_batch_size:],
                                                         self.input_y: y[batch*mini_batch_size:]})
                batch += 1

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.input_x: x})


if __name__ == '__main__':
    data = load_boston()
