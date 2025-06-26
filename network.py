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
    centered = center_matrix(matrix)  # centering digits like mnist does.
    flattened = [pixel for row in centered for pixel in row]
    forward(flattened, activations, weights, biases)
    return activations[3].index(max(activations[3]))

def center_matrix(matrix):
    total_mass = 0
    sum_x, sum_y = 0, 0

    for y in range(28):
        for x in range(28):
            val = matrix[y][x]
            total_mass += val
            sum_x += x * val
            sum_y += y * val

    if total_mass == 0:
        return matrix

    center_x = sum_x / total_mass
    center_y = sum_y / total_mass

    shift_x = int(round(14 - center_x))
    shift_y = int(round(14 - center_y))

    # Create new blank 28x28 matrix
    new_matrix = [[0.0 for _ in range(28)] for _ in range(28)]

    for y in range(28):
        for x in range(28):
            new_x = x + shift_x
            new_y = y + shift_y
            if 0 <= new_x < 28 and 0 <= new_y < 28:
                new_matrix[new_y][new_x] = matrix[y][x]

    return new_matrix