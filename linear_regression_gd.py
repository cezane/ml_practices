from numpy import *
import matplotlib.pyplot as plt
import time

def compute_err_for_points(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        # [x, y] from dataset (csv file)
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m*x + b))**2
        return total_error / float(len(points))

def step_gradient(current_b, current_m, points, learning_rate):
    #gradient descent
    gradient_b = 0
    gradient_m = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_b += -(2/N) * (y - ((current_m * x) + current_b))
        gradient_m += -(2/N) * x * (y - ((current_m * x) + current_b))
        new_b = current_b - (learning_rate * gradient_b)
        new_m = current_m - (learning_rate * gradient_m)
    
    return [new_b, new_m, (gradient_b**2 + gradient_m**2)**0.5]
    

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, acceptable_err):
    start_time = time.time()
    b = starting_b
    m = starting_m
    current_err = compute_err_for_points(b, m, points) #Stop condition based in RSS
    gradient_size = current_err
    i = 0
    errors = []
    while (acceptable_err <= current_err and acceptable_err <= gradient_size and i < num_iterations):
        b, m, gradient_size = step_gradient(b, m, array(points), learning_rate)
#        if (gradient_size <= acceptable_err):
#            break
            
        current_err = compute_err_for_points(b, m, points)
        errors.append(current_err)
        print("Gradient descent partial with new b = {}, new m = {}, gradient size = {} and error = {} at iteration {}." \
                                    .format(b, m, gradient_size, current_err, i+1))
        i += 1
    
    end_time = time.time()
    print("Execution time for gradient descent solution: {}.".format(end_time - start_time))

    plt.xlabel("Iterations")
    plt.ylabel("Errors")
    plt.plot(range(0, i), errors)
    return [b, m]

def closed_linear_reg(points):
    start_time = time.time()
    sum_x = sum_y = 0
    for i in range(0, len(points)):
        sum_x += points[i, 0]
        sum_y += points[i, 1]
    
    x_mean = sum_x / float(len(points))
    y_mean = sum_y / float(len(points))
    
    a = b = 0
    for i in range(0, len(points)):
        a += (points[i, 0] - x_mean) * (points[i, 1] - y_mean)
        b += (points[i, 0] - x_mean)**2
        
    m = a / b
    b = y_mean - m * x_mean
    end_time = time.time()
    print("Execution time for closed linear regression: {}.".format(end_time - start_time))
    return [b, m]

def run():
    points = genfromtxt('income.csv', delimiter=',')
    #hyperparameters: 
    #too short --> too slow to converge
    #too great --> never converge
    learning_rate = 0.0008
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 14000
    acceptable_err = 0.00005
    initial_err = compute_err_for_points(initial_b, initial_m, points)
    
    print("Starting gradient descent with b = {}, m = {} and error = {}".format(initial_b, initial_m, initial_err))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, acceptable_err)
    print("Gradient descent finished with b = {}, m = {} and error = {}".format(b, m, compute_err_for_points(b, m, points)))
    
    [b_reg, m_reg] = closed_linear_reg(points)
    print("Closed linear regression finished with b = {}, m = {} and error = {}".format(b_reg, m_reg, compute_err_for_points(b_reg, m_reg, points)))
    
if __name__ == '__main__':
    run()