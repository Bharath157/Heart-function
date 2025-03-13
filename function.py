import numpy as np
import matplotlib.pyplot as plt

# Define the heart shape function
def heart_function(x, y):
    return x**2 + (1.2*y - np.sqrt(np.abs(x)))**2 - 1.2  #Function to define the heart shape equation

# Generate a range of x and y values for plotting the heart shape

x = np.linspace(-1.5, 1.5, 400)  # x values from -1.5 to 1.5 with 400 points
y = np.linspace(-1.5, 1.5, 400)  # y values from -1.5 to 1.5 with 400 points

X, Y = np.meshgrid(x, y) ## Creates a 2D grid from x and y ranges

Z = heart_function(X, Y) #output of the equation

plt.figure(figsize=(6, 6))  # Create a square plot with size 6x6 inches
plt.contour(X, Y, Z, levels=[0], colors='red')  # Plot the contour where Z = 0
plt.title("Heart Shape Graph")  # Title of the plot
plt.xlabel("X-axis")  # Label for the x-axis
plt.ylabel("Y-axis")  # Label for the y-axis
plt.grid(True)  # Show the grid lines on the plot
plt.show()  # Display the plot
