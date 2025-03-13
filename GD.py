import numpy as np
def gradient_descent(x, y):
    m1_curr = m2_curr = 0  
    iterations = 10000
    n = len(x)
    learning_rate = 0.001
    for i in range(iterations):
        y_predicted = m1_curr * x + m2_curr 
        cost = (1/n) * sum((y - y_predicted) ** 2) 
        m1_d = -(2/n) * sum(x * (y - y_predicted))  
        m2_d = -(2/n) * sum(y - y_predicted)  
        m1_curr -= learning_rate * m1_d  
        m2_curr -= learning_rate * m2_d  
        if i % 1000 == 0:  
            print(f"Iteration {i}: m1 = {m1_curr}, m2 = {m2_curr}, cost = {cost}")
    return m1_curr, m2_curr
np.random.seed(42) 
x = np.array([1, 2, 3, 4, 5])
y = 2 * x + 3 + np.random.normal(0, 1, size=len(x))
m1_final, m2_final = gradient_descent(x, y)
print(f"Final values: m1 = {m1_final}, m2 = {m2_final}")