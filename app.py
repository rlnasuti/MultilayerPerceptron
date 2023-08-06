import numpy as np
from mlp import MLP

def save_weights_to_file(mlp, file_name):
    with open(file_name, 'w') as file:
        file.write("Weights from Input to Hidden Layer:\n")
        for row in mlp.weights1:
            file.write(' '.join(map(str, row)))
            file.write('\n')
            
        file.write("\nBias for Hidden Layer:\n")
        file.write(' '.join(map(str, mlp.bias1)))
        
        file.write("\n\nWeights from Hidden to Output Layer:\n")
        for row in mlp.weights2:
            file.write(' '.join(map(str, row)))
            file.write('\n')

        file.write("\nBias for Output Layer:\n")
        file.write(' '.join(map(str, mlp.bias2)))

# Training data for the XOR function
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

# Initialize MLP
mlp = MLP(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)

mlp.train(X=X, y=y, epoochs=100000)

# Check the initial prediction before training
initial_predictions = mlp.predict(X)
print(initial_predictions)

save_weights_to_file(mlp, 'weights.txt')
