from linear_models import LinearModel, LinearWithBasis, FourierBasis, mse, nll, sigmoid
from optimization import gd, sgd, least_squares_solution, fg_regression, sample_data, fg_classification, sgd_update, gd_earlystopping
from hw_linearfa import data_generator
import numpy as np
import matplotlib.pyplot as plt

def compute_optimal_eta(x, model):
    if isinstance(model, LinearWithBasis):
        feats = model.basis.predict(x)
    else:
        feats = x

    X = np.hstack([feats, np.ones((feats.shape[0],1))])
    C = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvals(C)
    eta_OPT = min((1.0 / (np.max(eigvals) + np.min(eigvals))).real, 1.0)
    return eta_OPT  # this should be close to the optimal step size

def data_split(D, ptest=0.3, pval=0.1):
    """
    Randomly split the data into training, validation, and testing sets
    :param D: data (X, y)
    :param ptest: fraction of data to use for testing
    :param pval: fraction of data to use for validation
    :return: training and testing data
    """
    X, y = D
    n = X.shape[0]  # number of data points
    ntest = int(n * ptest)  # number of testing points
    nval = int(n * pval)  # number of validation points
    ntrain = n - ntest - nval  # number of training points
    I = np.random.permutation(n)  
    X, y = X[I], y[I]  # shuffle the data so the classes are mixed up
    # TODO replace the code below with the correct code to split the data into training, validation, and testing sets
    Xtrain, ytrain = X[:2], y[:2]
    Xval, yval = X[2:4], y[2:4]
    Xtest, ytest = X[4:], y[4:]
    return (Xtrain, ytrain), (Xval, yval), (Xtest, ytest)
    
# plot a 2d version of the data to see how the classes are separated
def plot_data():
    np.random.seed(2)
    n = 1000
    X, y = data_generator(n, 2, noise1=1.0, noise2=1.0, shift=1.0)
    fig, axs = plt.subplots()
    axs.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Class 0')
    axs.plot(X[y == 1, 0], X[y == 1, 1], 'bx', label='Class 1')
    axs.set_xlabel('x1')
    axs.set_ylabel('x2')
    axs.legend()
    fig.savefig("classification_data_2d.png")



# create a plot showing that early stopping can prevent overfitting
def q1():
    np.random.seed(2)
    n = 50  # number of samples from each class. TODO: try different values for this parameter. 
    # Note that n being large or small relative to the number of features can have a big impact on the results.
    # You should try and make it so that the test loss gets worse over time. 
    # Update the p_test and p_val parameters accordingly. 

    d = 3  # number of features for the data. Don't change this. 
    noise1 = 1.0
    noise2 = 1.0
    shift = 0.5
    X, Y = data_generator(n, d, noise1=noise1, noise2=noise2, shift=shift)
    D = (X, Y)
    
    p_test = 0.2  # TODO find the optimal value for this parameter. Optimal value meaning the test loss is minimized.
    p_val = 0.6   # TODO find the optimal value for this parameter. 
    save_file_name = 'early_stopping_{0}.pdf'.format(n)
    Dtrain, Dval, Dtest = data_split(D, ptest=p_test, pval=p_val)
    print("Training data size: ", Dtrain[0].shape[0], "Validation data size: ", Dval[0].shape[0], "Testing data size: ", Dtest[0].shape[0])

    ranges = np.array([(-2 * noise1, max(2* noise1, shift + noise2*2)) for i in range(d)])
    order = 1  # TODO find the optimal value for this parameter
    basis = FourierBasis(d, order=order, ranges=ranges)
    
    # apply the basis function to the data so we don't have to compute it multiple times
    Dtrain = basis.predict(Dtrain[0]), Dtrain[1]  
    Dval = basis.predict(Dval[0]), Dval[1]
    Dtest = basis.predict(Dtest[0]), Dtest[1]
    
    # set up the model and optimization parameters
    model = LinearModel(basis.num_outputs())
    w0 = np.concatenate(model.params())
    max_iter = 10  # TODO make sure to set this value large enough to observe overfitting. 
    eta = 0.0  # TODO find a good value for this parameter
    eps = 1e-8  # TODO find a good value for this parameter
    fg = lambda w: fg_classification(w, model, Dtrain)  # function that computes the loss and gradient
    valf = lambda w: nll(Dval[1], sigmoid(model.predict(Dval[0])))  # compute the validation loss
    
    # run the optimization
    w, iter, train_loss, val_loss, weights = gd_earlystopping(fg, w0, valf, eta, max_iter, eps=eps, continue_after_early_stopping=True)
    
    test_loss = []
    for w in weights:
        model.set_params((w[:-1], w[-1]))
        yhat_test = sigmoid(model.predict(Dtest[0]))
        test_err = nll(Dtest[1], yhat_test)
        test_loss.append(test_err)

    fig, axs = plt.subplots()
    axs.plot(train_loss, label=r'$l_{D_{train}}(w)$')
    axs.plot(val_loss, label=r'$l_{D_{val}}(w)$')
    axs.plot(test_loss, label=r'$l_{D_{test}}(w)$')
    axs.vlines([iter], min(min(train_loss), min(val_loss), min(test_loss))*0.9, 
               max(max(train_loss), max(val_loss), max(test_loss))*1.1, linestyles='dashed')
    axs.set_xlabel('Iteration')
    axs.set_ylabel('Loss')
    axs.legend()
    fig.savefig(save_file_name)

"""
    Implement a function that finds the best model on the data set for the give hyperparameters
    This means you should create a model, train it on the training data. You should also create a validation set from the data to avoid overfitting. 
    The hyperparameters will be all values you need to create the model and train it. 
"""
def find_best_model(D, hyperparameters):
    # TODO: note that these hyperparameters are necessary ones you will need to assign in the sample hyperparameters function.
    # TODO: read through this code and make sure it makes sense. 
    
    # Set 1: create validation dataset
    pval = hyperparameters['pval']
    Dtrain, Dval, _ = data_split(D, ptest=0.0, pval=pval)

    # Step 2: create a model
    lower_range = hyperparameters['lower_range']
    upper_range = hyperparameters['upper_range']
    ranges = np.array([(lower_range, upper_range) for i in range(Dtrain[0].shape[1])])
    order = hyperparameters['order']
    basis = FourierBasis(Dtrain[0].shape[1], order=order, ranges=ranges)
    
    # Step 3: apply the basis function to the data so we don't have to compute it multiple times
    Dtrain = basis.predict(Dtrain[0]), Dtrain[1]
    Dval = basis.predict(Dval[0]), Dval[1]

    # Step 4: set up the model and optimization parameters
    model = LinearModel(basis.num_outputs())
    w0 = np.concatenate(model.params())
    max_iter = hyperparameters['max_iter']
    eta = hyperparameters['eta']
    eps = 1e-4
    fg = lambda w: fg_classification(w, model, Dtrain)  # function that computes the loss and gradient
    valf = lambda w: nll(Dval[1], sigmoid(model.predict(Dval[0])))  # compute the validation loss

    # Step 4: run the optimization  
    w, iter, train_loss, val_loss, weights = gd_earlystopping(fg, w0, valf, eta, max_iter, eps=eps, continue_after_early_stopping=False)

    model_final = LinearWithBasis(basis)    
    model_final.set_params((w[:-1], w[-1]))
    return model_final

def sample_hyperparameters():
    """
    Randomly sample hyperparameters to create and optimize a model. You can choose to do this however you want.
    """
    # you can sample an integer with np.random.randint(a, b) to get a random integer between a and b
    # you can sample a float with np.random.uniform(a, b) to get a random float between a and b
    # you can sample a value from a list with np.random.choice([a, b, c]) to get a random value from the list
    # you can also just set some values to be constants. These will not be optimized
    hyperparameters = {}
    hyperparameters['lower_range'] = -1.0  # this is the minimium value for the data. You shouldn't need to change this.
    hyperparameters['upper_range'] = 3.0  # this is the maximum value for the data. You shouldn't need to change this.
    hyperparameters['order'] = np.random.randint(1, 6)  # TODO sample this value
    hyperparameters['max_iter'] = 10  # TODO sample or set this value 
    hyperparameters['pval'] = 0.9 # TODO sample this value to optimize it. The value I put is not necessarily a good guess
    hyperparameters['eps'] = 1e-12  # TODO sample or set this value. This is the stopping criterion for the optimization. 
    hyperparameters['eta'] = 0  # TODO sample this value. This is the step size for the optimization. Note that the step size will need to change based on the number of features (order of the fourier basis)
    return hyperparameters

def sample_k_folds(D, K=5):
    """
    Randomly split the data into K folds of (D^i_{train}, D^i_{val}) pairs

    :param D: data (X, y)
    :param K: number of folds
    :return: list of K pairs of training and validation data
    """
    X, y = D
    n = X.shape[0]
    I = np.random.permutation(n)
    X, y = X[I], y[I]
    Dfolds = []  # TODO replace this with the correct code to split the data into K folds
    return Dfolds

def xval_model_selection(D, K=5, hyp_samples=10):
    """
    Perform cross-validation to select the best model hyperparameters and return the model trained on the full dataset with the best hyperparameters
    :param D: data (X, y)
    :param K: number of folds
    :param hype_sample: number of hyperparameter samples to try
    :return: best hyperparameters, best model
    """
    best_hyperparameters = None
    best_val_loss = np.inf
    Dfolds = sample_k_folds(D, K)
    hyp_sets = [sample_hyperparameters() for i in range(hyp_samples)]  # generate all hyperparameter sets
    for hyperparameters in hyp_sets:
        val_loss = 0
        # TODO: update val_loss to be the average validation loss over all the folds.
        # The model should be trained on the training data and evaluated on the validation data for each fold. 
        # You can use the find_best_model function to train the model. 

        if val_loss < best_val_loss:  # check if this is the best model so far
            best_val_loss = val_loss
            best_hyperparameters = hyperparameters
    final_model = find_best_model(D, best_hyperparameters)  # train the model on the full dataset
    return best_hyperparameters, final_model

HYPERS_FROM_Q1 = {'lower_range': -2, 'upper_range': 3, 'order': 1, 'max_iter': 10, 'eta': 0.0, 'pval': 0.9} # TODO use the hyperparameters you found in q1()
# run cross-validation to select the best hyperparameters, then retrain model on the full training set. 
# Compare the model performance to the hyperparameters chosen in q1()
def q2():
    np.random.seed(0)
    n = 100  # number of samples from each class. TODO: try different values for this parameter. Update the p_test and p_val parameters accordingly.
    d = 4
    noise1 = 1.0
    noise2 = 1.0
    shift = 1.0
    X, y = data_generator(n, d, noise1=noise1, noise2=noise2, shift=shift)
    D = (X, y)
    p_test = 0.3 
    
    Dtrain, _, Dtest = data_split(D, ptest=p_test, pval=0.0)  # split the data into training and testing sets
    print("Training data size: ", Dtrain[0].shape[0], "Testing data size: ", Dtest[0].shape[0])

    K = 2 # number of folds for cross-validation TODO: find the optimal value for this parameter
    num_hyps = 1  # number of hyperparameter samples to try TODO: find good value for this parameter that doesn't take too long to run
    
    # Step 1: find the best hyperparameters
    best_hyperparameters, model = xval_model_selection(Dtrain, K=K, hyp_samples=num_hyps)
    print("Best hyperparameters: ", best_hyperparameters)
    model_q2 = find_best_model(Dtrain, HYPERS_FROM_Q1)
    

    # Step 2: evaluate the model on the test set
    yhat_test = sigmoid(model.predict(Dtest[0]))
    test_loss = nll(Dtest[1], yhat_test)
    print("Test loss x-val: ", test_loss)
    yhat_test_q2 = sigmoid(model_q2.predict(Dtest[0]))
    test_loss_q2 = nll(Dtest[1], yhat_test_q2)
    print("Test loss q2: ", test_loss_q2)
    print("Difference in test loss: ", test_loss_q2 - test_loss)  # use this difference to answer the question. Positive number means xval was better. 

# repeat q2() with a different value but run the process multiple times to see how much variance there is the results of the model
def q3():
    np.random.seed(0)
    n = 100  
    d = 4
    noise1 = 1.0
    noise2 = 1.0
    shift = 1.0

    xval_test_losses = []
    q2_test_losses = []
    for i in range(50):  # run the data generation and model selection process 50 times
        X, y = data_generator(n, d, noise1=noise1, noise2=noise2, shift=shift)
        D = (X, y)
        Dtrain, _, Dtest = data_split(D, ptest=0.3, pval=0.0)

        K = 2 # number of folds for cross-validation TODO: use the value you found in q2()
        num_hyps = 1 # TODO use the value you found in q2()
        best_hyperparameters, model = xval_model_selection(Dtrain, K=K, hyp_samples=num_hyps)
        print("Best hyperparameters: ", best_hyperparameters)
        
        model_q2 = find_best_model(Dtrain, HYPERS_FROM_Q1)

        test_loss = nll(Dtest[1], sigmoid(model.predict(Dtest[0])))
        xval_test_losses.append(test_loss)
        test_loss_q2 = nll(Dtest[1], sigmoid(model_q2.predict(Dtest[0])))
        q2_test_losses.append(test_loss_q2)

    print("Average test loss x-val: ", np.mean(xval_test_losses))
    print("Average test loss q2: ", np.mean(q2_test_losses))
    print("Difference in test loss: ", np.mean(q2_test_losses) - np.mean(xval_test_losses))

    # plot the distributions of the test loss differences
    fig, axs = plt.subplots()
    quantiles = np.arange(0, 1, 0.02)
    differences = np.array(q2_test_losses) - np.array(xval_test_losses)
    axs.plot(quantiles, np.quantile(differences, quantiles), label=None)
    axs.legend()
    axs.set_xlabel('Quantile')
    axs.set_ylabel('Test loss differences')
    fig.savefig('loss_distribution.png')


if __name__ == '__main__':
    q1()  # TODO: uncomment to run question 1
    # q2()  # TODO: uncomment to run question 2
    # q3()  # TODO: uncomment to run question 3

    