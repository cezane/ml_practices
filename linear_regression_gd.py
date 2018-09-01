from numpy import *

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
    
    return [new_b, new_m]
    

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]

def run():
    points = genfromtxt('income.csv', delimiter=',')
    #hyperparameters: 
    #too short --> too slow to converge
    #too great --> never converge
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    initial_err = compute_err_for_points(initial_b, initial_m, points)
    print("Starting gradient descent with b = {}, m = {} and error = {}".format(initial_b, initial_m, initial_err))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("Gradient descent finished with b = {}, m = {} and error = {}".format(b, m, compute_err_for_points(b, m, points)))

if __name__ == '__main__':
    run()