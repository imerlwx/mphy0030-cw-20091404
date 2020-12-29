import numpy as np

## define a function to automatically compute gradient
def finite_difference_gradient(x, y, a, h, learning_rate):
    
    m = float(len(x)) # number of data sets
    k = float(len(a)) # number of parameters

    # for the cost function 'J' we know that its gradient about   
    # 'a[k]' is just (h[i] - y[i]) * x[k] / m
    # get the 'a' after gradient
    for k in range(0, k):
        for i in range(0, m):
            a[k] -= learning_rate * (h[i] - y[i]) * x[k] / m

    return a, m
    

## define a function to compute cost fuction and find its minimum
def gradient_descent(a_intial, x, y):
    
    # define basic parameter for gradient
    learning_rate = 0.0001
    iteration_num = 300
    tolerance = 0.001

    a = a_intial 
    h = x * a  # get intial a and h
    totalError = np.zeros((iteration_num, 1))  # set up error of gradient descent result

    n = 1 # intial iteration number

    # compute the cost function J(a) and iterate it
    while n <= iteration_num:

        a, m = finite_difference_gradient(x, y, a, h, learning_rate)

        h = x * a # update the multivariate function

        for i in range(0, m):
            totalError[n] += ((h[i] - y[i]) ** 2) / (2 * m) # compute the cost function result
    
        print(f"Iteration: {n}, totalError: {totalError[n]}")

        if totalError[n] - totalError[n-1] > tolerance:
            n += 1
        else:
            break

def quadratic_polynomial():
    