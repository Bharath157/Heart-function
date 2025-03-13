import numpy as np
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting

def gradient_descent(x, y):
    m1_curr = m2_curr = 0  # Initialize slope (m1) and intercept (m2) to 0
    iterations = 10000  # Number of iterations for gradient descent
    n = len(x)  # Number of data points
    learning_rate = 0.08  # Step size for updating parameters
    
    for i in range(iterations):
        y_predicted = m1_curr * x + m2_curr  # Compute predicted values (y = m1*x + m2)
        cost = (1/n) * sum((y - y_predicted) ** 2)  # Compute Mean Squared Error (MSE)
        
        # Compute gradients (partial derivatives)
        m1_d = -(2/n) * sum(x * (y - y_predicted))  # Gradient for m1
        m2_d = -(2/n) * sum(y - y_predicted)  # Gradient for m2
        
        # Update m1 and m2 using gradient descent formula
        m1_curr -= learning_rate * m1_d  
        m2_curr -= learning_rate * m2_d  
        
        # Print values every 1000 iterations to track progress
        if i % 1000 == 0:  
            print(f"Iteration {i}: m1 = {m1_curr}, m2 = {m2_curr}, cost = {cost}")
    
    return m1_curr, m2_curr

# Generate dataset with Gaussian noise
np.random.seed(42)  # Set seed for reproducibility
x = np.array([1, 2, 3, 4, 5])  # Input feature values
y = 2 * x + 3 + np.random.normal(0, 1, size=len(x))  # True relationship: y = 2x + 3 + noise

# Train the model using gradient descent
m1_final, m2_final = gradient_descent(x, y)
print(f"Final values: m1 = {m1_final}, m2 = {m2_final}")

# Plot the dataset and regression line
plt.scatter(x, y, color='red', label='Data Points')  # Scatter plot of data points
y_pred = m1_final * x + m2_final  # Compute predicted values using final m1 and m2
plt.plot(x, y_pred, color='blue', label='Regression Line')  # Plot regression line

# Label axes and add title
plt.xlabel('X (Feature)')  
plt.ylabel('Y (Target)')
plt.legend()  # Show legend
plt.title('Linear Regression using Gradient Descent')

# Display the plot
plt.show()
