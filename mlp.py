import numpy as np

# Defining the MLP class with a learning rate
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)
        self.bias1 = np.random.rand(self.hidden_size)
        self.bias2 = np.random.rand(self.output_size)

    def forward(self, X):
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output
    
    def hidden_activation(self, X):
        # Compute the activation of the hidden layer
        hidden_layer_activation = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        return hidden_layer_activation


    def backward(self, X, y, output):
        # Compute the gradients
        d_weights2 = np.dot(self.layer1.T, (2*(y - output) * self.sigmoid_derivative(output)))
        d_weights1 = np.dot(X.T, (np.dot(2*(y - output) * self.sigmoid_derivative(output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # Update the weights
        self.weights2 += self.learning_rate * d_weights2
        self.weights1 += self.learning_rate * d_weights1

        

    def train(self, X, y, epoochs):
        for i in range(epoochs):
            output = self.forward(X)
            self.backward(X, y, output)
        
    def predict(self, X):
        return self.forward(X)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
