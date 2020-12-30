import numpy as np
from matplotlib import pyplot as plt

## define a function to compute gradient by using finite fifference
## finite difference uses the value from packward and forward point
def finite_difference_gradient(f_plus, f_minus, dx):

    h = (f_plus - f_minus) / 2 * dx

    return h

## define a function to compute the minimum of a given polynomial
def gradient_descent(a, x):
    
    # define basic parameter for gradient
    step_size = 0.5
    iteration_num = 100
    tolerance = 0.00001
    dx = 0.1

    n = 1 # intial iteration number
    f = np.zeros((iteration_num, 1)) # set up intial f
    f[0] = a @ x

    # compute the cost function J(a) and iterate it
    while n < iteration_num:
        
        f_plus = a @ (x + dx) 
        f_minus = a @ (x - dx) 

        h = finite_difference_gradient(f_plus, f_minus, dx) # get gradient using finite_difference_gradient
        
        for k in range(len(x)):
            x[k, 0] += -step_size * h  # gradient descent

        f[n] = a @ x

        #print(f"Iteration: {n}, f(x): {f[n]}")
        print("Iteration: %d, f(x): %f" %(n, f[n]))

        # if the change of f is smaller than tolerance, then stop the iteration
        if f[n-1] - f[n] < tolerance:
            break
        else:
            n += 1

    #fig = plt.figure()
    m = np.arange(0, iteration_num).reshape(iteration_num, 1)
    plt.plot(m, f)
    plt.show()

    return f

## implement gradient descent to a quadratic polynomial 
def quadratic_polynomial():
    
    # set up intial a, x, f
    a = np.random.rand(1, 10)
    x_intial = np.array([[0.1], [0.1], [0.1]])

    x = np.array([x_intial[0] ** 2, x_intial[1] ** 2, x_intial[2] ** 3, 
                x_intial[0] * x_intial[1], x_intial[0] * x_intial[2], 
                x_intial[1] * x_intial[2], x_intial[0], x_intial[1], x_intial[2], [1.]]) 

    gradient_descent(a, x)

quadratic_polynomial()