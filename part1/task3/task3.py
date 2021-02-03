import numpy as np
from matplotlib import pyplot as plt

## define a function to compute gradient by using finite fifference
## finite difference uses the value from packward and forward point
def finite_difference_gradient(f_plus, f_minus, dx):

    h = (f_plus - f_minus) / 2 * dx

    return h

## define a polynomial
def quadratic_polynomial(a, x):

    poly = np.array([x[0] ** 2, x[1] ** 2, x[2] ** 3, x[0] * x[1],
                 x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], [1.]])  

    f = a @ poly
    
    return f

## define a function to compute the minimum of a given polynomial
def gradient_descent(a, x):
    
    # define basic parameter for gradient
    step_size = 0.1
    iteration_num = 1000
    tolerance = 0.0001
    dx = 0.1

    n = 1 # initial iteration number
    f = np.zeros((iteration_num, 1)) # set up intial f
    f[0] = quadratic_polynomial(a, x)

    # compute the cost function J(a) and iterate it
    while n < iteration_num:
        
        f_plus = quadratic_polynomial(a, x + dx)
        f_minus = quadratic_polynomial(a, x - dx)

        h = finite_difference_gradient(f_plus, f_minus, dx) # get gradient using finite_difference_gradient
        
        for k in range(len(x)):
            x[k, 0] += -step_size * h  # gradient descent
        
        f[n] = quadratic_polynomial(a, x)

        print("Iteration: %d, f(x): %f" %(n, f[n]))

        # if the change of f is smaller than tolerance, then stop the iteration
        if abs(f[n-1] - f[n]) < tolerance:
            print("It uses %d iterations to reach the minimum %f" %(n, f[n]))
            break
        else:
            n += 1

    return n, f
    
# randomly generate a and input a estimate of x
a = np.random.rand(1, 10)
x = np.array([[1.], [1.], [1.]])

n, f = gradient_descent(a, x)

# plot the change of f after n iterations to make sure it's decreasing
m = np.arange(0, n).reshape(n, 1)
plt.title("Change of f after n iterations")
plt.xlabel("iterations")
plt.ylabel("quadratic polynomial value")
plt.plot(m, f[0:n, 0])
#plt.show()
plt.savefig('task3.png')