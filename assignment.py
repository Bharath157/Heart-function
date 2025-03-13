import numpy as np

x = np.random.normal(9,4,10)
noise = np.random.normal(0,1,10)
m1_current = 0
m2_current = 0
y =  5 * x + 4

def gradient_descent(x,y,m1_current,m2_current):
    learning_rate = 0.001
    iterations = 10000
    n = len(x)
    for i in range(iterations):
        y_predicted = m1_current * x + m2_current 
        cost = (1/n) * sum((y - y_predicted) ** 2)
        m1_d = -(2/n) * sum(x * (y-y_predicted))
        m2_d = -(2/n) * sum(y-y_predicted)
        m1_current -= learning_rate * m1_d
        m2_current -= learning_rate * m2_d
        print(f"m1:{m1_current}, m2:{m2_current}, cost:{cost}, current_iteration:{i} ")
        
gradient_descent(x,y,m1_current,m2_current)