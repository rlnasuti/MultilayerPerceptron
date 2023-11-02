import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, activation_function=ActivationFunction.SIGMOID, use_adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        if activation_function not in ActivationFunction:
            raise ValueError("Unsupported activation function")
        self.activation_function = activation_function
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # He/Kaiming initialization for ReLU, otherwise use normal random initialization
        if self.activation_function == ActivationFunction.RELU:
            self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
            self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        else:
            self.weights1 = np.random.rand(self.input_size, self.hidden_size)
            self.weights2 = np.random.rand(self.hidden_size, self.output_size)

        self.bias1 = np.random.rand(self.hidden_size)
        self.bias2 = np.random.rand(self.output_size)

        if self.use_adam:
            # Initialize first and second moment variables for Adam
            self.m_weights1 = np.zeros_like(self.weights1)
            self.m_weights2 = np.zeros_like(self.weights2)
            self.m_bias1 = np.zeros_like(self.bias1)
            self.m_bias2 = np.zeros_like(self.bias2)

            self.v_weights1 = np.zeros_like(self.weights1)
            self.v_weights2 = np.zeros_like(self.weights2)
            self.v_bias1 = np.zeros_like(self.bias1)
            self.v_bias2 = np.zeros_like(self.bias2)

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

    def backward(self, X, y, output, t):
        # Error in the output layer
        output_error = y - output  # Shape: (number of samples, output_size)
        
        # Gradient for output layer biases
        d_bias2 = np.sum(output_error * self.activate_derivative(output), axis=0)
        
        # Error in the first hidden layer
        hidden_error = np.dot(output_error, self.weights2.T) * self.activate_derivative(self.layer1)  # Shape: (number of samples, hidden_size)
        
        # Gradient for first layer biases
        d_bias1 = np.sum(hidden_error, axis=0)

        # Compute the gradients for weights
        d_weights2 = np.dot(self.layer1.T, (2 * output_error * self.activate_derivative(output)))
        d_weights1 = np.dot(X.T, (2 * hidden_error * self.activate_derivative(self.layer1)))

        if self.use_adam:
            self.m_weights1 = self.beta1 * self.m_weights1 + (1 - self.beta1) * d_weights1
            self.m_weights2 = self.beta1 * self.m_weights2 + (1 - self.beta1) * d_weights2
            self.m_bias1 = self.beta1 * self.m_bias1 + (1 - self.beta1) * d_bias1
            self.m_bias2 = self.beta1 * self.m_bias2 + (1 - self.beta1) * d_bias2

            self.v_weights1 = self.beta2 * self.v_weights1 + (1 - self.beta2) * (d_weights1 ** 2)
            self.v_weights2 = self.beta2 * self.v_weights2 + (1 - self.beta2) * (d_weights2 ** 2)
            self.v_bias1 = self.beta2 * self.v_bias1 + (1 - self.beta2) * (d_bias1 ** 2)
            self.v_bias2 = self.beta2 * self.v_bias2 + (1 - self.beta2) * (d_bias2 ** 2)

            m_hat_weights1 = self.m_weights1 / (1 - self.beta1 ** t )
            m_hat_weights2 = self.m_weights2 / (1 - self.beta1 ** t )
            m_hat_bias1 = self.m_bias1 / (1 - self.beta1 ** t )
            m_hat_bias2 = self.m_bias2 / (1 - self.beta1 ** t )

            v_hat_weights1 = self.v_weights1 / (1 - self.beta2 ** t )
            v_hat_weights2 = self.v_weights2 / (1 - self.beta2 ** t )
            v_hat_bias1 = self.v_bias1 / (1 - self.beta2 ** t )
            v_hat_bias2 = self.v_bias2 / (1 - self.beta2 ** t )

            self.weights1 += self.learning_rate * m_hat_weights1 / (np.sqrt(v_hat_weights1) + self.epsilon)
            self.weights2 += self.learning_rate * m_hat_weights2 / (np.sqrt(v_hat_weights2) + self.epsilon)
            self.bias1 += self.learning_rate * m_hat_bias1 / (np.sqrt(v_hat_bias1) + self.epsilon)
            self.bias2 += self.learning_rate * m_hat_bias2 / (np.sqrt(v_hat_bias2) + self.epsilon)

    def train(self, X_train, y_train, X_val, y_val, epochs):
        training_accuracies = []
        validation_accuracies = []
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            # Training
            output_train = self.forward(X_train)
            self.backward(X_train, y_train, output_train, epoch+1)
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

