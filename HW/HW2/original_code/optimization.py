import numpy as np
import timeit

def sgd_update(w, grad, eta):
    """
    Perform a single update step of stochastic gradient descent (also works for gradient descent)
    :param params: np.ndarray current model parameters (as a vector)
    :param grad: np.ndarray gradient of the loss with respect to the weights
    :param eta: float step size
    :return: updated weights
    """
    # TODO implement the update step of SGD
    return w

def gd(fg, w0, eta=0.01, max_iter=1000, wtol=1e-6):
    """
    Perform gradient descent and terminate when the weight change is less than a tolerance or the maximum number of iterations is reached.
    :param fg: function that returns the loss and gradient
    :param w0: initial weights
    :param eta: step size
    :param max_iter: maximum number of iterations
    :param tol: weight tolerance for stopping criterion
    :return: tuple containing the estimated weights, the number of iterations, losses, and weights at each iteration
    """
    # TODO read an understand this code
    w = w0
    iter = 0
    losses = []
    weights = []
    while iter < max_iter:
        loss, grad = fg(w)  # compute loss and gradient
        losses.append(loss)  # store the loss for visualization later
        weights.append(np.copy(w))  # store the weights for visualization later
        w_new = sgd_update(w, grad, eta)  # update the weights
        if np.linalg.norm(w_new - w) < wtol:  # check if the change in weights is less than the tolerance
            break
        w = w_new
        iter += 1
    weights.append(np.copy(w))  # store the final weights
    loss = fg(w)[0]  # compute the loss at the final weights
    losses.append(loss)
    return w, iter, losses, weights

def sgd(fg, w0, sample_fun, eta=0.01, max_iter=1000):
    """
    Perform stochastic gradient descent
    :param fg: function that returns the loss and gradient
    :param w0: initial weights
    :param D: data
    :param eta: step size
    :param max_iter: maximum number of iterations
    :return: tuple containing the estimated weights and the number of iterations
    """
    # TODO read and understand this code
    w = w0
    iter = 0
    losses = []
    weights = []
    while iter < max_iter:
        data = sample_fun()  # sample a single data point for computing the gradient
        loss, grad = fg(w, data) # compute the loss and gradient
        losses.append(loss)  # store the loss for visualization later
        weights.append(np.copy(w))  # store the weights for visualization later
        w_new = sgd_update(w, grad, eta) # update the weights
        w = w_new  # set the new weights
        iter += 1
    weights.append(np.copy(w))
    return w, iter, losses, weights

def least_squares_solution(X, y):
    """
    Compute the least squares solution for a linear model with input data X and output data y
    :param X: input data (or features of data)
    :param y: regression targets
    :return: estimated weights (w, b)
    """
    # TODO read and understand this code
    X = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of ones for the bias term. This simulates the b term in the linear model
    sol = np.dot(np.linalg.pinv(X), y)  # compute the least squares solution to Ax = b using the pseudo-inverse
    w, b = sol[:-1], sol[-1]  # split the solution into the weights and bias

    return w, b


def fg_regression(params, model, D, times=None):
    """
    This is a helper function that computes the loss and gradient for the given parameters. 
    It converts the vector parameters into the weights and bias of the model. 
    It then maps gradients back into a single vector.

    :param params: vector of parameters
    :param model: linear model with basis
    :param D: (x,y) input data and output data
    :return: loss, gradient

    """
    x, y = D
    w,b = params[:-1], params[-1]  # parameters are stored as a single vector
    model.set_params((w,b))  # set the parameters of the model
    loss, gparams = model.regression_loss(x, y)  # compute the loss and gradient
    grad = np.concatenate([gparams[0].flatten(), gparams[1].flatten()])  # concatenate the gradient into a single vector
    if times is not None:
        times.append(timeit.default_timer())  # store the time for visualization
    return loss, grad


def fg_classification(params, model, D, times=None):
    """
    This is a helper function that computes the loss and gradient for the given parameters. 
    It converts the vector parameters into the weights and bias of the model. 
    It then maps gradients back into a single vector.

    :param params: vector of parameters
    :param model: linear model with basis
    :param D: (x,y) input data and output data
    :return: loss, gradient

    """
    x, y = D
    w,b = params[:-1], params[-1]  # parameters are stored as a single vector
    model.set_params((w,b))  # set the parameters of the model
    loss, gparams = model.binary_classification_loss(x, y)  # compute the loss and gradient
    grad = np.concatenate([gparams[0].flatten(), gparams[1].flatten()])  # concatenate the gradient into a single vector
    if times is not None:
        times.append(timeit.default_timer())  # store the time for visualization
    return loss, grad

def sample_data(D):
    """
    Sample a single data point from the data
    :param D: data (X, y), X is a matrix with data points as rows and y is a vector of labels
    :return: single data point (x,y)
    """
    X,y = D
    m = len(y)
    return X[1,:].reshape(1, -1), np.array([y[1]])  # TODO replace this with a random data point


def gd_earlystopping(fg, w0, valf, eta=0.01, max_iter=1000,eps=1e-4, continue_after_early_stopping=False):
    """
    Perform gradient descent with early stopping based on a validation set. 
    However, if continue_after_early_stopping is set to True, instead of actually stopping the optimization, let it run for the maximum number of iterations. 
    When early stopping says to terminate, save the weights and this iteration number. 
    We will use this information to plot the validation loss over time and see when early stopping would have stopped the optimization. 
    This will make it clear why early stopping is useful.

    If continue_after_early_stopping is set to False, then stop the optimization when early stopping is triggered.
    
    There are many possible early stopping criteria. For this implementation, we will use the following:
    - If the validation loss becomes more than epsilon worse than the best validation loss, then stop the optimization and return the weights at the best validation loss.
      This means you should store the weights at the best validation loss and the iteration number when this occurred.
    Note that this stopping criteria is different than what we covered in class. 
    This one should be a bit more robust because the validation loss is not gauranteed to decrease at every iteration. It may go up then go back down. 
    
    :param fg: function that returns the loss and gradient
    :param w0: initial weights
    :param valf: function that computes the validation loss for a given weight vector
    :param eta: step size
    :param max_iter: maximum number of iterations
    :param eps: tolerance for early stopping
    :return: tuple containing the estimated weights, the number of iterations, training losses, validation losses, and weights at each iteration
    """
    w = w0
    iter = 0
    losses = []
    weights = []
    val_losses = []
    w_stop = np.copy(w)
    iter_stop = max_iter
    min_val = np.inf
    while iter < max_iter:
        # TODO impelment gradient descent with early stopping
        loss, grad = 0, np.zeros_like(w)  # TODO replace this 
        w_new = np.copy(w)  # TODO replace this with the weight updated after the gradient step
        losses.append(loss)  # log the loss
        weights.append(np.copy(w))  # log the weights
        val_loss = 0 # replace this with the validation loss for w_new
        val_losses.append(val_loss)
        
        # TODO implement early stopping
        # code goes here
        
        
        w = np.copy(w_new)
        iter += 1
    # TODO make sure w_stop, and iter_stop are set to be the best weights and iteration number based on the validation loss. 
    return w_stop, iter_stop, losses, val_losses, weights
