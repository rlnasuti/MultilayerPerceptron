import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# Training data for the XOR function
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

# Generate a grid of input values
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
activation_maps = []
# Initialize MLP
mlp = MLP(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)

mlp.train(X=X, y=y, epoochs=100000)
# Compute the hidden layer activation for each point in the grid
for x in x_values:
    for y in y_values:
        hidden_activation = mlp.hidden_activation([x, y]) # Modify to extract hidden activation
        activation_maps.append(hidden_activation)

activation_maps = np.array(activation_maps).reshape(len(x_values), len(y_values), -1)

# Plot the activation maps for each hidden neuron
for i in range(activation_maps.shape[-1]):
    plt.imshow(activation_maps[:, :, i], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', interpolation='nearest')
    plt.title(f'Activation Map for Hidden Neuron {i+1}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar(label='Activation')
    plt.show()
