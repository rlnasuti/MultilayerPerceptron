import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the forward pass using the provided weights and biases
def forward_pass(x1, x2):
    h1 = sigmoid(5.57 * x1 - 3.80 * x2 + 0.82)
    h2 = sigmoid(-4.34 * x1 + 7.30 * x2 + 0.35)
    h3 = sigmoid(5.30 * x1 + 5.14 * x2 + 0.41)
    h4 = sigmoid(5.40 * x1 - 3.26 * x2 + 0.36)
    h5 = sigmoid(0.66 * x1 - 1.14 * x2 + 0.75)
    output = sigmoid(-6.38 * h1 - 10.92 * h2 + 15.98 * h3 - 4.80 * h4 - 0.71 * h5 + 0.37)
    return output

# Generate a grid of points in the input space
x1_vals = np.linspace(0, 1, 100)
x2_vals = np.linspace(0, 1, 100)
output_grid = np.zeros((100, 100))

# Evaluate the network's output over the grid
for i, x1 in enumerate(x1_vals):
    for j, x2 in enumerate(x2_vals):
        output_grid[i, j] = forward_pass(x1, x2)

# Plot the XOR data points
plt.scatter([0, 1], [0, 1], c='blue', label='0')
plt.scatter([0, 1], [1, 0], c='red', label='1')

# Plot the decision boundary
plt.contourf(x1_vals, x2_vals, output_grid.T, levels=50, alpha=0.6, cmap='RdBu_r')
plt.colorbar(label='Network Output')
plt.contour(x1_vals, x2_vals, output_grid.T, levels=[0.5], colors='black', linewidths=1)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Decision Boundary and XOR Data Points')
plt.show()
