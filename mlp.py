import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, activation_function=ActivationFunction.SIGMOID, use_momentum=False, momentum_factor=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        if activation_function not in ActivationFunction:
            raise ValueError("Unsupported activation function")
        self.activation_function = activation_function

        # He/Kaiming initialization for ReLU, otherwise use normal random initialization
        if self.activation_function == ActivationFunction.RELU:
            self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
            self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        else:
            self.weights1 = np.random.rand(self.input_size, self.hidden_size)
            self.weights2 = np.random.rand(self.hidden_size, self.output_size)

        self.bias1 = np.random.rand(self.hidden_size)
        self.bias2 = np.random.rand(self.output_size)

        # Initialize momentum variables
        if self.use_momentum:
            self.momentum_weights1 = np.zeros_like(self.weights1)
            self.momentum_weights2 = np.zeros_like(self.weights2)
            self.momentum_bias1 = np.zeros_like(self.bias1)
            self.momentum_bias2 = np.zeros_like(self.bias2)

    def forward(self, X):
        self.layer1 = self.activate(np.dot(X, self.weights1) + self.bias1)
        self.output = self.activate(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output
    
    def activate(self, x):
        if self.activation_function == ActivationFunction.SIGMOID:
            return self.sigmoid(x)
        elif self.activation_function == ActivationFunction.TANH:
            return self.tanh(x)
        elif self.activation_function == ActivationFunction.RELU:
            return self.relu(x)
        else:
            raise ValueError("Unsupported activation function")
        
    def activate_derivative(self, x):
        if self.activation_function == ActivationFunction.SIGMOID:
            return self.sigmoid_derivative(x)
        elif self.activation_function == ActivationFunction.TANH:
            return self.tanh_derivative(x)
        elif self.activation_function == ActivationFunction.RELU:
            return self.relu_derivative(x)
        else:
            raise ValueError("Unsupported activation function")
    
    def hidden_activation(self, X):
        return self.activate(np.dot(X, self.weights1) + self.bias1)

    def backward(self, X, y, output):
        # Compute the gradients
        d_weights2 = np.dot(self.layer1.T, (2 * (y - output) * self.activate_derivative(output)))
        d_weights1 = np.dot(X.T, (np.dot(2 * (y - output) * self.activate_derivative(output), self.weights2.T) * self.activate_derivative(self.layer1)))

        # Update the weights with momentum
        if self.use_momentum:
            self.momentum_weights1 = self.momentum_factor * self.momentum_weights1 + self.learning_rate * d_weights1
            self.weights1 += self.momentum_weights1
            self.momentum_weights2 = self.momentum_factor * self.momentum_weights2 + self.learning_rate * d_weights2
            self.weights2 += self.momentum_weights2
        else:
            self.weights1 += self.learning_rate * d_weights1
            self.weights2 += self.learning_rate * d_weights2

    def train(self, X_train, y_train, X_val, y_val, epochs):
        training_accuracies = []
        validation_accuracies = []
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            # Training
            output_train = self.forward(X_train)
            self.backward(X_train, y_train, output_train)
            train_accuracy = self.calculate_accuracy(output_train, y_train)
            train_loss = self.calculate_loss(output_train, y_train)
            training_accuracies.append(train_accuracy)
            training_losses.append(train_loss)

            # Validation
            output_val = self.forward(X_val)
            val_accuracy = self.calculate_accuracy(output_val, y_val)
            val_loss = self.calculate_loss(output_val, y_val)
            validation_accuracies.append(val_accuracy)
            validation_losses.append(val_loss)

        return training_accuracies, validation_accuracies, training_losses, validation_losses

    def calculate_accuracy(self, output, y):
        predicted_labels = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def calculate_loss(self, output, y):
        """
        Calculate Mean Squared Error Loss
        """
        return np.mean((output - y) ** 2)
        
    def predict(self, X):
        return self.forward(X)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        # Limit the values of x to avoid overflow
        x = np.clip(x, -20, 20)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def tanh_derivative(self, x):
        # Limit the values of x to avoid overflow
        clipped_x = np.clip(x, -20, 20)
        return 1.0 - self.tanh(clipped_x)**2

    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

