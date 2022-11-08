import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self.assign_weights(np.zeros(n_features))
        for iterator in range(self.max_iter):
            gradient_sum = np.zeros(n_features)
            for (each_x, each_y) in zip(X, y):
                gradient_sum = np.add(gradient_sum,self._gradient(each_x, each_y))
            gradient_sum = gradient_sum/n_samples
            self.W += self.learning_rate * (-1) * (gradient_sum)

            ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))
        for iterator in range(self.max_iter):
            randomised_indices = np.arange(n_samples)
            np.random.shuffle(randomised_indices)

            for batch_index in range(0, n_samples, batch_size):
                each_step = n_samples - batch_index if batch_index + batch_size > n_samples else batch_size
                gradient_sum = np.zeros(n_features)

                iterating_batch_indices = randomised_indices[batch_index:batch_index + each_step]
                for each_index in iterating_batch_indices: #in zip(X[batch_index:batch_index + each_step], y[batch_index:batch_index + each_step]):
                    each_x, each_y = X[each_index],y[each_index]
                    gradient_sum = np.add(gradient_sum,self._gradient(each_x, each_y))
                gradient_sum = gradient_sum/each_step
                self.W += self.learning_rate * (-1) * (gradient_sum)

            ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))
        for iterator in range(self.max_iter):
            randomised_indices = np.arange(n_samples)
            np.random.shuffle(randomised_indices)
            for i in randomised_indices:
                each_x,each_y = X[i],y[i]
                _gradient = self._gradient(each_x,each_y)
                self.W += self.learning_rate * (-1) * (_gradient)

            ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """

		### YOUR CODE HERE
        exp_value = np.exp(_y *np.dot(_x,self.W))
        _value = (-1*_y)/(1+exp_value)

        _g = np.array([each_x * _value for each_x in _x])
        return _g

    ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE
        n_samples, _ = X.shape
        predict_proba = np.zeros((n_samples,2))
        for i,each_x in enumerate(X):
            _val = np.exp(-1 * np.dot(each_x, self.W))
            predict_proba[i][0] = 1 / (1 + _val)
            predict_proba[i][1] = 1-predict_proba[i][0]
        return predict_proba

        ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE

        output = []
        probabilities = self.predict_proba(X)
        n_samples, _ = X.shape

        for i in range(n_samples):
            if probabilities[i][0]>=0.5:
                output.append(1)
            else:
                output.append(-1)
        return np.array(output)
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        predicted_output = self.predict(X)
        n_samples, _ = X.shape
        score = 0
        for i in range(n_samples):
            if predicted_output[i]==y[i]:
                score = score+1
        return score/n_samples
		### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self

