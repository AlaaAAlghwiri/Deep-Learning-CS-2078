import timeit
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def mse(y, y_hat):
    """
    Compute the mean squared error (scaled by 0.5)
    :param y: np.ndarray of shape (m,) representing the target values, where m is the number of data points
    :param y_hat: np.ndarray of shape (m,) representing the predicted values
    :return: mean squared error
    """
    return 0.5 * np.mean((y - y_hat)**2) #

def mse_gradient(y, y_hat):
    """
    Compute the mean squared error and gradient of the mean squared error with respect to the predictions y_hat
    :param y: np.ndarray of shape (m,) representing the target values, where m is the number of data points
    :param y_hat: np.ndarray of shape (m,) representing the predicted values
    :return: gradient of mean squared error with respect to predictions
    """
    mse_loss = mse(y, y_hat)
    grad = (y_hat - y) / y.shape[0] 
    return mse_loss, grad

def nll(y, y_hat):
    """
    Compute the average negative log likelihood of the data
    :param y: np.ndarray of shape (m,) representing the {0,1} class labels, where m is the number of data points
    :param y_hat: np.ndarray of shape (m,) representing the pre-sigmoid predicted probabilities of each label
    :return: negative log likelihood
    """
    prob = sigmoid(y_hat)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)) # TODO replace with correct implementation


def nll_gradient(y, y_hat):
    """
    Compute the average negative log likelihood and the gradient of the negative log likelihood with respect to the predictions y_hat
    :param y: np.ndarray of shape (m,) representing the {0,1} class labels, where m is the number of data points
    :param y_hat: np.ndarray of shape (m) representing the pre-sigmoid predicted probabilities of each label
    :return: negative log likelihood, gradient of negative log likelihood with respect to predictions
    """
    prob = sigmoid(y_hat)
    nll_loss = nll(y, y_hat)
    grad = (prob - y) / y.shape[0] 
    return nll_loss, grad
  

def accuracy(y, y_hat):
    """
    Compute the average negative log likelihood of the data
    :param y: vector of binar {0,1} class labels
    :param y_hat: vector of predicted probabilities of class 1
    :return: negative log likelihood
    """
    acc = np.mean(y == (y_hat > 0.5))
    return acc

def sigmoid(x):
    """
    Compute the sigmoid function \sigma(x) = \frac{1}{1 + e^{-x}}
    :param x: input could be a scalar or a numpy array
    :return: sigmoid of x
    """
    return  1/(1 + np.exp(-x)) # TODO replace with correct implementation

class LinearModel(object):
    def __init__(self, num_features):  # TODO read this small code block so you know what the variables are
        self.w = np.zeros(num_features)  
        self.b = np.zeros(1)  # a vector of length 1. This makes it easier to compute outputs of multiple inputs at once

    def params(self):
        """
        Return the weights
        :return: weights
        """
        return (self.w, self.b)
    
    def set_params(self, params):
        """
        Set the weights
        :param params: tuple of (w,b)
        """
        self.w = params[0]
        self.b[:] = params[1]

    def predict(self, x):
        """
        Predict the output given the input. This should work for a batch of data and a single data point.
        :param x: input data of shape (m, n) where m is the number of data points and n is the number of features
        :return: output data of shape (m,)
        """
        assert len(x.shape) == 2
        out = np.dot(x, self.w) + self.b # TODO replace with correct implementation
        assert out.shape == (x.shape[0],), "Expected shape {}, got shape {}".format((x.shape[0],), out.shape)
        return out
    
    def predict_with_gradient(self, x):
        """
        Predict the output and return the gradient of the output with respect to the weights
        :param x: input data of shape (m, n) where m is the number of data points and n is the number of features
        :return: output data of shape (m,), gradient of output with respect to weights (w,b)
        """
        assert len(x.shape) == 2
        gradw = x  # gradient of output with respect to weights  TODO replace with correct implementation
        gradb = np.ones(x.shape[0])  # gradient of output with respect to bias  TODO replace with correct implementation
        return self.predict(x), (gradw, gradb)

    def regression_loss(self, x, y):
        """
        Compute the loss and gradient of the loss with respect to the weights
        :param w: weights
        :param x: input data
        :param y: output data
        :return: loss, gradient of loss with respect to weights
        """
        # TODO: replace this code with correct implementation
        y_hat = self.predict(x)
        loss, grad_loss = mse_gradient(y, y_hat)
        gw = np.mean(grad_loss[:, np.newaxis] * x, axis=0)  #should be (num_features,)
        gb = np.mean(grad_loss) # should be (1,)  
        assert gw.shape == self.w.shape, "Expected shape {}, got shape {}".format(self.w.shape, gw.shape)
        assert gb.shape == self.b.shape, "Expected shape {}, got shape {}".format(self.b.shape, gb.shape)
        return loss, (gw, gb)

    def binary_classification_loss(self, x, y):
        """
        Compute the loss and gradient of the loss with respect to the weights
        :param w: weights
        :param x: input data
        :param y: output data
        :return: loss, gradient of loss with respect to weights
        """

        y_hat = self.predict(x)
        loss, grad_loss = nll_gradient(y, y_hat) # should be scalar
        gw = np.mean(grad_loss[:, np.newaxis] * x, axis=0) #should be (num_features,)
        gb = np.mean(grad_loss)  # should be (1,)
        assert gw.shape == self.w.shape, "Expected shape {}, got shape {}".format(self.w.shape, gw.shape)
        assert gb.shape == self.b.shape, "Expected shape {}, got shape {}".format(self.b.shape, gb.shape)
        return loss, (gw, gb)

def make_C_matrix(num_inputs, order):
    """
    Create the matrix C for the Fourier basis function. Each row of C is a combination of the Fourier basis function.
    Each entry of C contains the pi times the frequency to apply a given input. The ith row of C will be used to create 
    the feature \sin(\pi o_{i,1} x_1 + \pi o_{i,2} x_2 + ... + \pi o_{i,n} x_n) where o_{i,j} is an order of frequency 
    between 0 and order. Each possible combination of frequencies will be used to create the basis functions.

    :param num_inputs: number of input dimensions
    :param order: order of the Fourier basis function
    :return: np.array C of shape ((order+1)^num_inputs, num_inputs)
    """
    combos = list(it.product(range(order+1), repeat=num_inputs))
    C = np.zeros((len(combos), num_inputs))
    for i, combo in enumerate(combos):
        C[i] = combo
    C *= np.pi
    return C

class FourierBasis(object):
    def __init__(self, num_inputs, order, ranges):
        """
        Create a Fourier basis function. The basis function will be a combination of sin and cos of the frequencies of the inputs.
        
        :param num_inputs: number of input dimensions
        :param order: order of the Fourier basis function
        :param ranges: np.array of shape (num_inputs, 2) where the first column is the minimum value of the input and the second column is the maximum value of the input
        """
        self.num_inputs = num_inputs
        self.order = order
        self.ranges = np.array(ranges)
        self.C = make_C_matrix(num_inputs, order)
    
    def predict(self, x):
        """
        create the basis function features for the given the input. You should first normalize the data to be between 0 and 1 based on the ranges of the features.
        :param x: np.ndarray containing input data of shape (m, n) where m is the number of data points and n is the number of inputs
        :return: np.ndarray contraining features of shape (m, 2 (order+1)^n) where the first half of features are sin and the second half are cos
        """
        assert len(x.shape) == 2, "Expected shape (m, {}), got shape {}".format(self.num_inputs, x.shape)
        assert x.shape[1] == self.num_inputs, "Expected shape (m, {}), got shape {}".format(self.num_inputs, x.shape)
        xnormed = (x - self.ranges[:, 0]) / (self.ranges[:, 1] - self.ranges[:, 0]) # TODO: normalize the data to be between 0 and 1 based on the ranges of each feature
        argument = np.dot(xnormed, self.C.T) * np.pi
        sin_feats = np.sin(argument)  # TODO: compute the sin features using xnormed
        cos_feats = np.cos(argument)  # TODO: compute the cos features using xnormed
        feats = np.concatenate((sin_feats, cos_feats), axis=1)  # Combine the features into a single array
        assert feats.shape == (x.shape[0], 2 * self.C.shape[0])
        return feats
        
    
    def num_outputs(self):
        """
        Return the number of basis functions
        :return: number of basis functions
        """
        return 2 * self.C.shape[0]

class LinearWithBasis(object):
    """
    Linear model with a basis function. This class is a wrapper for the basis function and linear model classes.
    """
    def __init__(self, basis):
        self.basis = basis
        self.linear_model = LinearModel(basis.num_outputs())

    def params(self):
        """
        Return the weights
        :return: weights
        """
        return self.linear_model.params()
    
    def set_params(self, params):
        """
        Set the weights
        :param params: tuple of (w,b)
        """
        self.linear_model.set_params(params)
        

    def predict(self, x):
        """
        Predict the output given the input. This should work for a batch of data and a single data point.
        :param x: input data
        :return: output data
        """
        features = self.basis.predict(x)  # TODO: see this is just a simple composition of the basis fuction and linear model
        return self.linear_model.predict(features)
    
    def predict_with_gradient(self, x):
        """
        Predict the output and return the gradient of the output with respect to the weights
        :param x: input data
        :return: output data, gradient of output with respect to weights (w,b)
        """
        features = self.basis.predict(x) # features are (m, basis.num_outputs())
        y_hat, (gradw, gradb) = self.linear_model.predict_with_gradient(features)
        # gradw is (m, basis.num_outputs())
        # gradb is (m,)  
        return y_hat, (gradw, gradb)

    def regression_loss(self, x, y):
        """
        Compute the loss and gradient of the loss with respect to the weights
        :param w: weights
        :param x: input data
        :param y: output data
        :return: loss, gradient of loss with respect to weights
        """
        features = self.basis.predict(x)  
        return self.linear_model.regression_loss(features, y)
    
    def binary_classification_loss(self, x, y):
        """
        Compute the loss and gradient of the loss with respect to the weights
        :param w: weights
        :param x: input data
        :param y: output data
        :return: loss, gradient of loss with respect to weights
        """
        features = self.basis.predict(x)
        return self.linear_model.binary_classification_loss(features, y)

    


if __name__ == '__main__':
    # TODO test out the above functions here. I have provided some test cases, but they may not cover all edge cases.

    yhats = np.array([1, 0, 3, 4])
    y = np.array([1, 1, 1, 3])
    print("MSE", mse(y, yhats))
    assert mse(y, yhats) == 0.75

    mgrad = mse_gradient(y, yhats)
    print("MSE gradient", mgrad[0], mgrad[1])
    assert mgrad[0] == mse(y, yhats), "loss from mse_gradient is not consistent with mse"
    assert np.allclose(mgrad[1], np.array([0, -0.25, 0.5, 0.25])), "gradient for mse is incorrect"

    y = np.array([1, 0, 1, 0])
    target_probs = np.array([0.9, 0.1, 0.8, 0.2])
    yhats = -np.log((1 / target_probs) - 1)
    assert np.allclose(target_probs, sigmoid(yhats)), "sigmoid function is incorrect: expected {}, got {}".format(target_probs, sigmoid(yhats))

    print("NLL", nll(y, yhats)) 
    loss = -(np.log(0.9) + np.log(1-0.1) + np.log(0.8) + np.log(1-0.2))/4
    assert np.allclose(nll(y, yhats), loss, atol=1e-6), "nll is incorrect: expected {}, got {}".format(loss, nll(y, yhats))

    ngrad = nll_gradient(y, yhats)
    print("NLL gradient", ngrad[0], ngrad[1])
    assert ngrad[0] == nll(y, yhats), "loss from nll_gradient is not consistent with nll"
    assert np.allclose(ngrad[1], -np.array([0.1/4, -0.1/4, 0.2/4, -0.2/4])), "gradient for nll is incorrect"


    C = make_C_matrix(2, 3)
    print("C matrix for fourier basis")
    print(C / np.pi)  # TODO the entries of C have every possible combination of frequencies between 0 and 3

    basis = FourierBasis(2, 3, [(0, 1), (0, 1)])
    x = np.array([[0, 0], [np.pi, 0], [np.pi, np.pi]])
    feats = basis.predict(x)
    assert feats.shape == (3,basis.num_outputs()), "expected shape (3, {}), got shape {}".format(basis.num_outputs,feats.shape)
    print("Features for input x")
    print(feats)  # TODO: the features are the sin and cos of the frequencies. Check to make sure outputs are as expected. Cosine of 0 is 1, sin of 0 is 0, etc.
                
