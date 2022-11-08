#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""


class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k

    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        nsamples, nfeatures = X.shape
        # creating one-hot encoding vectors for each sample and initialising them as y matrix
        y = np.zeros((nsamples,self.k))
        for i,each_label in enumerate(labels):
            y[i][int(each_label)] = 1

        self.assign_weights(np.zeros((self.k, nfeatures))) # reverse??

        # the following code is similar to the fit_miniBGD code made for logistic regression
        # just that the definition of y and the gradient used here is different.
        # hence the code inside the gradient function would be different .
        # also the dimension of gradient_sum will also be different. same dimension as weight vector
        for iterator in range(self.max_iter):
            for batch_index in range(0, nsamples, batch_size):
                each_step = nsamples - batch_index if batch_index + batch_size > nsamples else batch_size
                gradient_sum = np.zeros((self.k, nfeatures)) ## reverse??

                for each_x,each_y in zip(X[batch_index:batch_index + each_step], y[batch_index:batch_index + each_step]):
                    gradient_sum = np.add(gradient_sum,self._gradient(each_x, each_y))
                gradient_sum = gradient_sum/each_step
                self.W += self.learning_rate * (-1) * (gradient_sum)

    ### END YOUR CODE

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        signal = list(map(lambda _k: np.dot(_x, self.W[_k]), range(self.k)))

        softmax_probabilities = self.softmax(signal)
        sub = softmax_probabilities - _y
        grad = np.dot(sub.reshape(-1, 1), _x.reshape(1, -1))
        return grad

    ### END YOUR CODE

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""

        ### You must implement softmax by youself, otherwise you will not get credits for this part.
        # ### YOUR CODE HERE

        return np.exp(x) / np.sum(np.exp(x))
        #return output

    ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE

        probability_list = np.array([self.softmax(np.matmul(self.W, _x)) for _x in X])
        prediction = np.argmax(probability_list,axis=1)
        return prediction

    ### END YOUR CODE

    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE
        predicted_labels = self.predict(X)
        nsamples,_ = X.shape
        score = 0
        for i in range(nsamples):
            if labels[i]==predicted_labels[i]:
                score+=1
        return score/nsamples

    ### END YOUR CODE
    def assign_weights(self, weights):
        self.W = weights
        return self

