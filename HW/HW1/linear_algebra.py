import timeit
import numpy as np
import matplotlib.pyplot as plt

def mat_vec_mul(A, x):
    """
    Multiply a matrix by a vector using for loops
    :param A: matrix
    :param x: vector
    :return: vector representing matrix-vector product
    """
    n = len(A)
    m = len(A[0])
    y = [0] * n  # TODO replace with correct implementation
    for i in range(n):         
        for j in range(m):     
            y[i] += A[i][j] * x[j]
    return y

def mat_mat_mul(A, B):
    """
    Multiply two matrices using for loops
    :param A: matrix
    :param B: matrix
    :return: matrix-matrix product
    """
    Arows = len(A)
    Acols = len(A[0])
    Brows = len(B)
    Bcols= len(B[0])
    if Acols!=Brows:
        print ("Product is not defined as the # of columns in the first matrix is not equal to the number of rows in the second matrix")
    else:
        C = [[ 0 for i in range(Bcols)] for j in range(Arows)]
        for i in range(Arows):
            for j in range(Bcols):
                total = 0
                for ii in range(Acols):
                    total += A[i][ii] * B[ii][j]
                    C[i][j] = total
     # TODO replace with correct implementation
    return C

def dot_product(x, y):
    """
    Compute the dot product of two vectors using for loops
    :param x: vector
    :param y: vector
    :return: scalar representing the dot product of x and y
    """
    z = sum(i*j for i,j in zip(x,y))  # TODO replace with correct implementation
    return z

def linear_algebra_timings():
    """
    Time the matrix-vector, matrix-matrix, and dot product operations 
    for vectors and matricies with dimension n in [1,2,4,8,16,32,64,128,256,512,1024]. 
    """
    n_values = [2**i for i in range(10)]  # You can make this larger but it can take a long time. 
    mat_vec_times_fl = []
    mat_mat_times_fl = []
    dot_times_fl = []

    mat_vec_times_np = []
    mat_mat_times_np = []
    dot_times_np = []

    for n in n_values:
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        B = np.random.rand(n, n)
        y = np.random.rand(n)
        
        # mat-vec multiplication
        start = timeit.default_timer()
        mat_vec_mul(A, x) # TODO replace with your code for matrix-vector multiplication of A and x here
        mat_vec_times_fl.append(timeit.default_timer() - start)
        start = timeit.default_timer()
        np.multiply(A, x) # TODO replace with numpy code for matrix-vector multiplication of A and x here
        mat_vec_times_np.append(timeit.default_timer() - start)

        # mat-mat multiplication
        start = timeit.default_timer()
        mat_mat_mul(A, B) # TODO replace with your code for matrix-matrix multiplication of A and B here
        mat_mat_times_fl.append(timeit.default_timer() - start)
        start = timeit.default_timer()
        np.multiply(A,B) # TODO replace with numpy code for matrix-matrix multiplication of A and B here
        mat_mat_times_np.append(timeit.default_timer() - start)

        # dot product
        start = timeit.default_timer()
        dot_product(x, y) # TODO replace with your code for dot product of x and y here
        dot_times_fl.append(timeit.default_timer() - start)
        start = timeit.default_timer()
        np.dot(x,y) # TODO replace with numpy code for dot product of x and y 
        dot_times_np.append(timeit.default_timer() - start)

    return n_values, (mat_vec_times_fl, mat_mat_times_fl, dot_times_fl), (mat_vec_times_np, mat_mat_times_np, dot_times_np)
    

if __name__ == '__main__':
    n_values, fl_times, np_times = linear_algebra_timings()
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)
    axs[0].plot(n_values, fl_times[0], label='For Loop')
    axs[0].plot(n_values, np_times[0], label='Numpy')
    axs[0].set_title('Matrix-Vector Multiplication')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].legend()

    axs[1].plot(n_values, fl_times[1], label='For Loop')
    axs[1].plot(n_values, np_times[1], label='Numpy')
    axs[1].set_title('Matrix-Matrix Multiplication')
    axs[1].set_xlabel('n')
    axs[1].set_ylabel('Time (s)')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')

    axs[2].plot(n_values, fl_times[2], label='For Loop')
    axs[2].plot(n_values, np_times[2], label='Numpy')
    axs[2].set_title('Dot Product')
    axs[2].set_xlabel('n')
    axs[2].set_ylabel('Time (s)')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('linear_algebra.png')
