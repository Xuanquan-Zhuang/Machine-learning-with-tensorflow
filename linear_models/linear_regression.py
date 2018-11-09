#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Messiah
# @Date  : 18-9-21

import tensorflow as tf
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self, intercept=True, learning_rate=0.001):
        self.has_intercept = intercept
        self.learning_rate = learning_rate
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.float32, shape=[None, None])
            self.input_y = tf.placeholder(tf.float32, shape=[None, 1])
            self.test_x = tf.placeholder(tf.float32, shape=[None, None])

            self.feature_num = tf.placeholder(tf.int16, shape=[1])

            self.initializer = None
            self.coefficients = None
            self.intercept = None
            self.prediction = None
            self.loss = None
            self.train = None

        self.sess = tf.Session(graph=self.graph)

    def fit(self, x, y, batch_size=100, epochs=1000):
        with self.graph.as_default():
            self.coefficients = tf.Variable(tf.truncated_normal([x.shape[1], 1]), dtype=tf.float32)
            if self.has_intercept:
                self.intercept = tf.Variable(tf.truncated_normal([1]), dtype=tf.float32)
                self.prediction = tf.add(tf.matmul(self.input_x, self.coefficients), self.intercept)
            else:
                self.prediction = tf.matmul(self.input_x, self.coefficients)

            self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.prediction, self.input_y), 2))
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.initializer = tf.global_variables_initializer()
            self.sess.run(self.initializer)

        size = len(x)
        mini_batch_size = batch_size if batch_size < size else size

        for epoch in range(epochs):
            batch = 0
            while batch * mini_batch_size < size:
                if (batch + 1) * mini_batch_size <= size:
                    _, loss = self.sess.run([self.train, self.loss],
                                            feed_dict={self.input_x: x[batch*mini_batch_size:(batch+1)*mini_batch_size],
                                                       self.input_y: y[batch*mini_batch_size:(batch+1)*mini_batch_size].reshape([-1, 1])})
                else:
                    _, loss = self.sess.run([self.train, self.loss],
                                            feed_dict={self.input_x: x[batch*mini_batch_size:],
                                                       self.input_y: y[batch*mini_batch_size:].reshape([-1, 1])})
                batch += 1

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.input_x: x})


if __name__ == '__main__':
    data = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.2, random_state=1)

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)

    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    mse = ((y_test_pred - y_test.reshape([-1, 1])) ** 2).mean()

    print('Mean square error: ', mse)

    model_sklearn = linear_model.LinearRegression()
    model_sklearn.fit(x_train, y_train)

    y_test_pred_sklearn = model_sklearn.predict(x_test)
    mse_sklearn = ((y_test_pred_sklearn - y_test) ** 2).mean()

    print('Mean square error for sklearn:, ', mse_sklearn)
