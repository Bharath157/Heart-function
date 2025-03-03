import numpy as np  
import matplotlib.pyplot as plt  

# Function to perform gradient descent for the  linear regression
def gradient_descent(input_x, actual_y):
    slope_m1 = 0  #Initialize the slope m1 as 0
    intercept_m2 = 0   #  initialize the intercept as  0
    total_iterations = 10000  # Number of iterations 
    num_samples = len(input_x)  # Number of data points
    learning_rate = 0.0001  # Step size
    cost_values = []  # List to store cost function values
    
    
    for iteration in range(total_iterations):
        predicted_y = slope_m1 * input_x + intercept_m2  # calculating the predicted values using current parameters
        cost = (1 / num_samples) * sum((actual_y - predicted_y) ** 2)  # calculating the  MSE
        cost_values.append(cost)  # Store the cost value
        
        # calculating the Compute gradients 
        gradient_m1 = -(2 / num_samples) * sum(input_x * (actual_y - predicted_y))  # Gradient for slope m1
        gradient_m2 = -(2 / num_samples) * sum(actual_y - predicted_y)  # Gradient for intercept m2
        
        # Update the parameters using gradient descent formula
        slope_m1 -= learning_rate * gradient_m1  
        intercept_m2 -= learning_rate * gradient_m2  
        
        # Print values every 1000 iterations
        if iteration % 1000 == 0:  
            print(f"Iteration {iteration}: slope_m1 = {slope_m1}, intercept_m2 = {intercept_m2}, cost = {cost}")
    
    return slope_m1, intercept_m2, cost_values  # Return final slope, intercept, and cost history

# Generate dataset with Gaussian noise
np.random.seed(42)  # Set the  seed
input_x = np.array([1, 2, 3, 4, 5])  # Input  values
actual_y = 2 * input_x + 3 + np.random.normal(0, 1, size=len(input_x))  #  y = 2x + 3 + noise

final_slope_m1, final_intercept_m2, cost_values = gradient_descent(input_x, actual_y)
print(f"Final values: slope_m1 = {final_slope_m1}, intercept_m2 = {final_intercept_m2}")

# Plot the dataset and regression line
plt.figure(figsize=(10, 5))  # Set figure size

# Subplot 1: Scatter plot of data points and regression line
plt.subplot(1, 2, 1)  # Create subplot 
plt.scatter(input_x, actual_y, color='red', label='Data Points')  # Scatter plot of data points
predicted_y_values = final_slope_m1 * input_x + final_intercept_m2  # Calculating the predicted values using final parameters
plt.plot(input_x, predicted_y_values, color='blue', label='Regression Line')  # Plot the regression line

# Label axes and add title
plt.xlabel('Input X (Feature)')  
plt.ylabel('Actual Y (Target)')
plt.legend()  # Show legend
plt.title('Linear Regression using Gradient Descent')

# Subplot 2: Cost function over iterations
plt.subplot(1, 2, 2)  # Create subplot 
plt.plot(range(len(cost_values)), cost_values, color='green')  # Plot cost values over iterations
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Over Iterations')

plt.tight_layout()  
plt.show()  # Show plots
