import numpy as np

## define a function to automatically compute gradient
def finite_difference_gradient(x, a, h, learning_rate):
    
    m = float(len(x)) # number of data sets
    k = float(len(a)) # number of parameters

    # for the cost function 'J' we know that its gradient about   
    # 'a[k]' is just (h[i] - y[i]) * x[k] / m
    # get the 'a' after gradient
    for k in range(0, k):
        for i in range(0, m):
            a[k] -= learning_rate * (h[i] - y[i]) * x[k] / m

    return a
    

## define a function to compute cost fuction and find its minimum
def gradient_descent(a_intial, x, y, h):
    
    # define basic parameter for gradient
    learning_rate = 0.0001
    iteration_num = 300
    tolerance = 0.001
    
    n = 0 # intial iteration number

    a = a_intial 

    # compute the cost function J(a) and iterate it
    while n < iteration_num:

        a = finite_difference_gradient(x, a, h, learning_rate)

        h = x * a # update the multivariate function

        for i in range(0, len(h)):
            totalError += (h[i] - y[i]) ** 2
    
        n += 1

        if 

    return totalError / (2 * m)
