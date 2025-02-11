from linear_models import LinearModel, LinearWithBasis, FourierBasis, mse
from optimization import gd, sgd, least_squares_solution, fg_regression, sample_data, fg_classification, sgd_update
import numpy as np
import matplotlib.pyplot as plt
import timeit
import time

def regression_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * x**2 - x - 1

def generate_regression_data(n=100, noise=0.25):
    x = np.clip(np.random.randn(n),-2,2)  # sample inputs from a standard normal distribution (clipped to be in [-2,2])
    y = regression_function(x) + np.random.randn(n) * noise  # add noise to the output
    X = x.reshape(-1, 1)  # reshape into a matrix of (n,1)
    return X, y

def plot_regression_data(x, y, save_prefix='data'):
    plt.figure()
    plt.scatter(x, y, marker='o', color="black", s=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression Data')
    plt.savefig('{0}.png'.format(save_prefix))

def plot_all_lines(x, y, models, names, save_prefix='lines'):
    plt.figure()
    plt.scatter(x, y, marker='o', label=None, s=4, color="black")
    x = np.linspace(np.min(x), np.max(x), 200).reshape(-1, 1)
    colors = ["darkorange", "crimson", "dodgerblue"]
    for (model,name, color) in zip(models, names, colors):
        yhat = model.predict(x)
        style = '--' if 'LS' in name else '-'
        plt.plot(x, yhat, label=name, linestyle=style, color=color, linewidth=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.savefig('{0}.pdf'.format(save_prefix))

def plot_learning_curve(losses, weights, save_prefix='learning_curve'):
    weight_diffs = np.diff(weights, axis=0)
    wnorms = np.linalg.norm(weight_diffs, axis=1)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(losses)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Loss')
    axs[1].plot(wnorms)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Weight Change')
    plt.xlabel('Iteration')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
    plt.savefig('{0}.png'.format(save_prefix))

def data_generator(n, d, noise1=1.0, noise2=1.0, shift=1.0):
    """
    Generate data for a linear classifier with two classes
    :param n: number of samples from each class
    :param d: number of features
    :param noise1: noise level for class 1
    :param noise2: noise level for class 2
    :param shift: shift between the two classes
    """
    X1 = np.clip(np.random.randn(n, d), -2, 2) * noise1
    X2 = np.clip(np.random.randn(n, d), -2, 2) * noise2 + shift
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    # shuffle the data
    I = np.random.permutation(2 * n)
    X, y = X[I, :], y[I]
    return X, y


def q1():
    np.random.seed(0)  # make it so the randomness starts the same each time we run the code
    x, y = generate_regression_data(n=400)
    plot_regression_data(x, y)  # uncomment to plot just the data
    
    w,b = (0,0)  # TODO replace this line with one to compute the least squares solution
    model1 = LinearModel(1)
    model1.set_params((w, b))  # create a linear model with the estimated weights
    
    # TODO: Note that the hyperparameters I specified are not good values. You will need to try different values to get the desired fit.
    ranges = np.array([[-2, 2]])    # specify the range of the input data
    order = 0                       # TODO: specify the order for the basis function function
    num_inputs = 1                  # specify the number of input features
    basis = FourierBasis(num_inputs, order, ranges=ranges)  # create a radial basis function basis
    model2 = LinearWithBasis(basis)  # create a linear model with the fourier basis
    
    fg = lambda params: fg_regression(params, model2, (x, y))  # TODO: understand this function. This is a function that returns the loss and gradient for the specified parameters
    
    # perform gradient descent to find the optimal weights
    eta = 0.0  # TODO: specify the step size. 
    w0 = np.concatenate(model2.params())  # specify the initial weights
    max_iters = 10  # TODO specify the maximum number of iterations
    wtol = 0 # TODO specify the tolerance for the stopping criterion
    beta, iter, losses, weights = gd(fg, w0, eta=eta, max_iter=max_iters, wtol=wtol)

    model2.set_params((beta[:-1], beta[-1]))
    print("iter: ", iter, "loss: ", losses[-1])
    plot_learning_curve(losses, weights, save_prefix='fourier_learning_curve')

    # find optimal weights for the Fourier basis using least squares
    feats = model2.basis.predict(x)
    w,b = least_squares_solution(feats, y)  # compute the least squares solution for the features
    model3 = LinearWithBasis(basis)
    model3.set_params((w,b))  # create a linear model with the estimated weights
    
    plot_all_lines(x, y, [model1, model3, model2], ['Linear', 'Fourier-LS', 'Fourier'], save_prefix='all_lines')

# run sgd to train a linear classifier with 2 parameters
# plot the parameters of the weights over time to see them bounce around the optimum. 
# use multiple step sizes to see how the step size affects the convergence.
def q2():
    etas = [0.0, 0.0, 0.0]  # TODO replace this with a list of step sizes to try. They NEED to be in decreasing order for the plot to show up correctly. 
    max_iter = 10 # TODO find a value big enough to see the full convergence. SGD is fast so this can be BIG
    n = 100  # number of samples from each class. Don't change this for the plot you turn in, but you can change it for your own interests. 
    d = 2  # number of dimensions for the data. Don't change this or the code won't work. 
    X, y = data_generator(n, d, shift = 0.5)  # Generate the data
    ws = []  # list of weights over the optimization process
    fig, axs = plt.subplots()
    for eta in etas:
        model = LinearModel(d)  # create the model
        w0 = np.concatenate(model.params()) * 0 # make the initial weights zero. The multiplication by 0 is probably uncessary unless you change the initialization function of the class. The weights are also made into a single vector.
        fg = lambda w, D: fg_classification(w, model, D)  # this is the function that computes the loss and gradient
    
        w, iter, losses, weights = sgd(fg, w0, lambda: sample_data((X,y)), eta, max_iter)  # run the optimization using SGD
        axs.plot(np.array(weights)[:, 0], np.array(weights)[:, 1], 'o-', markersize=3, alpha=0.5, label=r'$\eta=${}'.format(eta))  # plot the path of the weights w. Note the bias term is also changing, but we aren't showing it in this plot. 
    
    axs.set_xlabel(r'$w_1$')
    axs.set_ylabel(r'$w_2$')
    axs.set_title('SGD path')
    axs.legend()
    fig.savefig('sgd_path.png')  # change this name to save the plot to a different file.
    
def regression_function2(x, coeffs1, coeffs2):
    y = np.sin(2 * np.pi * x * coeffs1) + 0.5 * (x * coeffs2)**2 - x - 1
    if len(y.shape) == 1 and coeffs1.shape[0] == 1:
        y = y[:,np.newaxis]
    return y.sum(axis=1)

def generate_regression_data2(n=100, k = 4, noise=0.25):
    """
    Generate data for a regression problem with multiple inputs
    
    :param n: number of data points
    :param k: number of input features
    :param noise: standard deviation of the noise
    :return: input data and output data
    """
    x = np.clip(np.random.randn(n, k),-2,2)  # sample inputs from a standard normal distribution (clipped to be in [-2,2])
    coeffs1 = np.random.randn(k)  # generate random coefficients for the regression function
    coeffs2 = np.random.randn(k)  # generate random coefficients for the regression function
    y = regression_function2(x, coeffs1, coeffs2) + np.random.randn(n) * noise  # add noise to the output
    x0 = np.random.rand(10_000, k) * 4 - 2
    fx = regression_function2(x0, coeffs1, coeffs2)
    y = y / np.sqrt(np.std(fx)**2 + noise**2)  # scale the output
    return x, y

def compare_optimizers(n=1000, order=3, k=4, eta=0.1, max_iters=50_000):
    wtol = 0.0
    x, y = generate_regression_data2(n=n, k=k)
    num_inputs = x.shape[1]
    ranges = np.ones((num_inputs,2))
    ranges[:,0] = -2
    ranges[:,1] = 2
    basis = FourierBasis(num_inputs, order, ranges=ranges)
    num_feats = basis.num_outputs()
    x = basis.predict(x)
    model = LinearModel(num_feats)


    time.sleep(0.2)
    # perform gradient descent
    gd_times = []
    w0 = np.concatenate(model.params()) * 0
    eta_gd = 1 / (num_feats/2 + 1)
    fg = lambda params: fg_regression(params, model, (x, y), times=gd_times)
    start2 = timeit.default_timer()
    beta, iter, losses2, weights2 = gd(fg, w0, eta=eta_gd, max_iter=max_iters, wtol=wtol)
    stop2 = timeit.default_timer()
    print("GD time: ", stop2 - start2, "iter: ", iter)

    time.sleep(0.2)
    # perform stochastic gradient descent
    sgd_times = []
    w0 = np.concatenate(model.params()) * 0
    fg_sgd = lambda w, D: fg_regression(w, model, D, times=sgd_times)
    eta_sgd = eta_gd * eta
    print("Optimal eta: ", eta_gd, "eta: ", eta, "eta_sgd: ", eta_sgd)
    
    start3 = timeit.default_timer()
    beta, iter, _, weights3 = sgd(fg_sgd, w0, lambda: sample_data((x,y)), eta=eta_sgd, max_iter=max_iters)
    stop3 = timeit.default_timer()
    print("SGD time: ", stop3 - start3, "iter: ", iter)

    losses3 = []
    for w in weights3:
        model.set_params((w[:-1], w[-1]))
        yhat = model.predict(x)
        lossw = mse(y, yhat) # need to compute the losses on the full dataset for each weight
        losses3.append(lossw)

    time.sleep(0.2)
    # compute least squares solution
    start = timeit.default_timer()
    # feats = model.basis.predict(x)
    w,b = least_squares_solution(x, y)
    # w,b = least_squares_solution(x, y)
    stop = timeit.default_timer()
    print("LS time: ", stop - start)
    model.set_params((w,b))
    loss1 = model.regression_loss(x, y)[0]
    
    gd_times = np.array(gd_times) - start2
    sgd_times = np.array(sgd_times) - start3
    return (gd_times, np.array(losses2)), (sgd_times, np.array(losses3[:-1])), (stop - start, loss1)

def average_opt_times(n=1000, order=4, eta=0.1, max_iters=50_000, num_trials=10):
    print("optimizing with n: {0}, order: {1}, eta: {2}, max_iters: {3}".format(n, order, eta, max_iters))
    gds = []
    sgds = []
    lss = []
    for i in range(num_trials):
        gd, sgd, ls = compare_optimizers(n=n, order=order, eta=eta, max_iters=max_iters)
        gds.append(gd)
        sgds.append(sgd)
        lss.append(ls)
    gd_times = np.mean([gd[0] for gd in gds], axis=0)
    sgd_times = np.mean([sgd[0] for sgd in sgds], axis=0)
    ls_times = np.mean([ls[0] for ls in lss], axis=0)
    gd_losses = np.mean([gd[1] for gd in gds], axis=0)
    sgd_losses = np.mean([sgd[1] for sgd in sgds], axis=0)
    ls_losses = np.mean([ls[1] for ls in lss], axis=0)
    return (gd_times, gd_losses), (sgd_times, sgd_losses), (ls_times, ls_losses)


def q3():
    # compare the times for SGD, GD, and LS for different numbers of data points
    # Each optimization approach will take a different amount of time to reach a low level of loss
    # The amount of time it will take will deepend (mostly) on the number of data points, the number of basis functions, and the step size
    # TODO: You will need to find the settings of these hyperparameters to show when each method is best. 
    # If you cannot find settings where each method is best, report the closest setting you found, and explain why you might not be able to find better.
    np.random.seed(0)
    fig, axs = plt.subplots(3,1, sharex=False, sharey=False, figsize=(6, 6))
    n1 = 2
    eta1 = 0.0  # this is a scaling factor for the step size for SGD. We will compute the step size for gradient descent as (1 / (num_features/2 +1) then scale that step size by this value for SGD. SGD_step_size = eta * step_size_for_gd
    max_iters1 = 2
    order1 = 1
    
    gd_1, sgd_1, ls_1 = average_opt_times(n=n1, order=order1, eta=eta1, max_iters=max_iters1, num_trials=10)
    axs[0].plot(gd_1[0], gd_1[1], label='GD')
    axs[0].plot(sgd_1[0], sgd_1[1], label='SGD')
    axs[0].scatter([ls_1[0]], [ls_1[1]], color='black', label='LS')
    axs[0].legend()

    n2 = 2
    eta2 = 0.0
    max_iters2 = 2
    order2 = 1

    gd_2, sgd_2, ls_2 = average_opt_times(n=n2, order=order2, eta=eta2, max_iters=max_iters2, num_trials=10)
    axs[1].plot(gd_2[0], gd_2[1], label='GD')
    axs[1].plot(sgd_2[0], sgd_2[1], label='SGD')
    axs[1].scatter([ls_2[0]], [ls_2[1]], color='black', label='LS')

    n3 = 2
    eta3 = 0.0
    max_iters3 = 2
    order3 = 1

    gd_3, sgd_3, ls_3 = average_opt_times(n=n3, order=order3, eta=eta3, max_iters=max_iters3, num_trials=10)
    axs[2].plot(gd_3[0], gd_3[1], label='GD')
    axs[2].plot(sgd_3[0], sgd_3[1], label='SGD')
    axs[2].scatter([ls_3[0]], [ls_3[1]], color='black', label='LS')
    
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[2].set_xlabel('Time (s)')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Loss')
    axs[2].set_ylabel('Loss')

    ranges = np.array([(-2, 2) for i in range(2)])
    num_feats1 = FourierBasis(order1, 4, ranges).num_outputs()
    num_feats2 = FourierBasis(order2, 4, ranges).num_outputs()
    num_feats3 = FourierBasis(order3, 4, ranges).num_outputs()

    axs[0].set_title('n={0}, features={1}'.format(n1, num_feats1))
    axs[1].set_title('n={0}, features={1}'.format(n2, num_feats2))
    axs[2].set_title('n={0}, features={1}'.format(n3, num_feats3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.5)
    fig.savefig('gd_sgd_times.png')

def compute_optimal_eta(x, model):
    """
    Compute the optimal step size for gradient descent for the given model and data. 
    The optimal step size should be 1/(max(eigvals) + min(eigvals)) of the covariance matrix of the features.
    NOTE: You should tune the step size manually before looking at this value. This is to understand a little bit about how to tune the step size.
    Using the optimal step size makes it easier to do some of the comparision between the different optimization methods.
    """
    if isinstance(model, LinearWithBasis):
        feats = model.basis.predict(x)
    else:
        feats = x

    X = np.hstack([feats, np.ones((feats.shape[0],1))])
    C = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvals(C)
    eta_OPT = min((1.0 / (np.max(eigvals) + np.min(eigvals))).real, 1.0) * 0.5
    return eta_OPT  # this should be close to the optimal step size


class MyBasis(object):
    def __init__(self, num_features):
        # TODO: implement whatever initialization you need
        # The function is: np.sin(2 * np.pi * x) + 0.5 * x**2 - x - 1 + noise, see regression_function
        pass

    def predict(self, x):
        """
        create the basis function features for the given the input.
        :param x: np.darray of shape (m,1) where m is the number of data points
        :return: output features
        """
        assert len(x.shape) == 2
        assert x.shape[1] == 1
        feats = np.ones(1)  # TODO replace this with your implementation of the basis function
        assert feats.shape == (x.shape[0], self.num_outputs())
        return feats
    
    def num_outputs(self):
        """
        Return the number of basis functions
        :return: number of basis functions
        """
        return 1  # TODO replace with the number of features created by your basis function

# compare using gradient descent with the basis function you created to using the fourier basis
def q4():
    np.random.seed(1)  
    fbflosses = []
    mybasislosses = []
    fbf_errors = []
    mybasis_errors = []
    x0 = np.linspace(-2,2, num=10_000).reshape(-1, 1)
    fx = regression_function(x0) # generate ground truth data points on the f(x) we want to model
    
    ns = [5, 10, 100, 400]  # test at different sample sizes
    for n in ns:
        x, y = generate_regression_data(n=n)
        # plot_regression_data(x, y)  # uncomment to plot just the data
        x = x.reshape(-1, 1)
        ranges = np.array([[-2, 2]])       
        order = 1  # TODO specify the order for the basis function function
        num_inputs = 1                
        basis1 = FourierBasis(num_inputs, order, ranges=ranges)  # create a radial basis function basis
        model1 = LinearWithBasis(basis1)  # create a linear model with the fourier basis
        
        basis2 = MyBasis(num_inputs)
        model2 = LinearWithBasis(basis2)

        fg1 = lambda params: fg_regression(params, model1, (x, y))  

        max_iters = 100   # TODO you might need to update this value
        wtol = 1e-4  # TODO you might need to update this value 

        # perform gradient descent to find the optimal weights
        eta = compute_optimal_eta(x, model1)
        w0 = np.concatenate(model1.params())  # specify the initial weights
        beta, iter, losses, weights = gd(fg1, w0, eta=eta, max_iter=max_iters, wtol=wtol)
        model1.set_params((beta[:-1], beta[-1]))

        fg2 = lambda params: fg_regression(params, model2, (x, y))  
        eta = compute_optimal_eta(x, model2)
        w0 = np.concatenate(model2.params())  # specify the initial weights
        beta, iter, losses2, weights2 = gd(fg2, w0, eta=eta, max_iter=max_iters, wtol=wtol)

        fbe = []
        for w in weights:
            model1.set_params((w[:-1], w[-1]))
            yhat = model1.predict(x0)
            error = np.mean((yhat - fx)**2)
            fbe.append(error)

        mbe = []
        for w in weights2:
            model2.set_params((w[:-1], w[-1]))
            yhat = model2.predict(x0)
            error = np.mean((yhat - fx)**2)
            mbe.append(error)

        fbf_errors.append(fbe)
        mybasis_errors.append(mbe)

        fbflosses.append(losses)
        mybasislosses.append(losses2)
        print("n: ", n, "fourier loss: ", np.round(losses[-1],4), "mybasis loss: ", np.round(losses2[-1],4), "fourier error: ", np.round(fbe[-1],4), "mybasis error: ", np.round(mbe[-1], 4))
    
    fig, axs = plt.subplots(2,2, sharex=True)
    colors = ["darkorange", "crimson", "dodgerblue", "darkgreen", "purple", "darkgrey"]
    for (i,n) in enumerate(ns):
        axs[0,0].plot(fbflosses[i], label='Fourier n={0}'.format(n), linestyle="solid", color=colors[i])
        axs[0,1].plot(mybasislosses[i], label='MyBasis n={0}'.format(n), linestyle="solid", color=colors[i])
        axs[1,0].plot(fbf_errors[i], label='Fourier n={0}'.format(n), linestyle="solid", color=colors[i])
        axs[1,1].plot(mybasis_errors[i], label='MyBasis n={0}'.format(n), linestyle="solid", color=colors[i])

    axs[0,0].legend()
    axs[0,1].legend()
    axs[0,0].set_ylabel('Loss on Data')
    axs[1,0].set_ylabel(r"Error on $f(x)$")
    axs[1,0].set_xlabel('Iteration')
    axs[1,1].set_xlabel('Iteration')
    axs[0,0].set_title("Fourier Basis")
    axs[0,1].set_title("MyBasis")    
    plt.savefig('fourier_vs_mybasis.png')


if __name__ == '__main__':
    q1()  # TODO: uncomment to run question 1
    q2()  # TODO: uncomment to run question 2
    q3()  # TODO: uncomment to run question 3
    q4()  # TODO: uncomment to run question 3