import json  # for importing the weights and biases.
from train_network import activations, forward, softmax

# activations for each layer
activations = [
    [0.0 for _ in range(784)],  # input layer
    [0.0 for _ in range(16)],  # hidden layer 1
    [0.0 for _ in range(16)],  # hidden layer 2
    [0.0 for _ in range(10)]  # output layer
]

with open('./models/model_9_accuracy_93.40.json', 'r') as file:
    model = json.load(file)
    weights = model["weights"]
    biases = model["biases"]

def guess(matrix):
    input_layer = [val for row in matrix for val in row]
    forward(input_layer, activations, weights, biases)
    return (activations[3].index(max(activations[3])))
