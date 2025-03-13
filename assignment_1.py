import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 * x + 3 + np.random.randn(100, 1)  # y = 2x + 3 + noise

# Initialize parameters
m1, m2 = 0.0, 0.0
learning_rate = 0.1
iterations = 1000

# Gradient Descent
for i in range(iterations):
    # Predicted y
    y_pred = m1 * x + m2
    
    # Compute gradients
    dm1 = (-2/len(x)) * np.sum(x * (y - y_pred))
    dm2 = (-2/len(x)) * np.sum(y - y_pred)
    
    # Update parameters
    m1 -= learning_rate * dm1
    m2 -= learning_rate * dm2
    
    # Print loss every 100 iterations
    if i % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Iteration {i}, Loss: {loss}, m1: {m1}, m2: {m2}")

print(f"Final parameters: m1 = {m1}, m2 = {m2}")

# Plot the data and the fitted line
plt.scatter(x, y, color='blue', label='Data points')  # Plot the data points
plt.plot(x, m1 * x + m2, color='red', label='Fitted line')  # Plot the fitted line
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent: Linear Regression')
plt.legend()
plt.show()