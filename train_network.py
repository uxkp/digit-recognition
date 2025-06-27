import random  # for initialising weights
from math import sqrt, exp
import json  # for exporting weights/biases.

# layer sizes
input_size = 784  # 28 x 28 (number of pixels on the digit grid)
hidden1_size = 16
hidden2_size = 16
output_size = 10  # 1 to 9

# activations for each layer
activations = [
    [0.0 for _ in range(input_size)],  # input layer
    [0.0 for _ in range(hidden1_size)],  # hidden layer 1
    [0.0 for _ in range(hidden2_size)],  # hidden layer 2
    [0.0 for _ in range(output_size)]  # output layer
]

# z-values (pre-activations)
zs = [
    [0.0 for _ in range(hidden1_size)],  # z for hidden layer 1
    [0.0 for _ in range(hidden2_size)],  # z for hidden layer 2
    [0.0 for _ in range(output_size)]  # z for output layer
]

# He initialisation of weights (good for ReLU neural nets)
weights = [
    [  # input to hidden layer 1 
        [random.gauss(0, sqrt(2 / input_size)) for _ in range(input_size)]
        for _ in range(hidden1_size)
    ],
    [  # hidden layer 1 to hidden layer 2
        [random.gauss(0, sqrt(2 / hidden1_size)) for _ in range(hidden1_size)]
        for _ in range(hidden2_size)
    ],
    [  # hidden layer 2 to output layer
        [random.gauss(0, sqrt(2 / hidden2_size)) for _ in range(hidden2_size)]
        for _ in range(output_size)
    ]
]

# initialising biases
biases = [
    [0.0 for _ in range(hidden1_size)],  # for hidden layer 1
    [0.0 for _ in range(hidden2_size)],  # for hidden layer 2
    [0.0 for _ in range(output_size)]    # for output layer
]

# gradient initialisation
delta_weights = [
    [[0.0 for _ in range(len(weights[i][0]))] for _ in range(len(weights[i]))]
    for i in range(3)
]

delta_biases = [
    [0.0 for _ in biases[i]]
    for i in range(3)
]

# ReLU (rectified linear unit) activation function
def relu(x): return max(0, x)

# derivative of ReLU(x)
def relu_prime(x): return 0 if x <= 0 else 1

# loading the mnist data.
def load_data(filename):
    data = []
    with open(filename, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            label = int(parts[0])
            pixels = [int(x) / 255.0 for x in parts[1:]]  # normalize
            data.append((pixels, label))
    if filename == "./data/mnist_train.csv":
        print("Training data loaded!")
    else:
        print("Testing data loaded!")
    return data

# gradient descent
def gradient_descent(training_data, test_data, epochs, learning_rate):
    for epoch in range(epochs):
        random.shuffle(training_data)  # shuffle samples

        for input_layer, label in training_data:
            forward(input_layer, activations, weights, biases)  # updates activations/zs
            backpropogation(target_list(label))  # computes gradient
            update_weights(learning_rate)  # applies gradient

        accuracy = (evaluate(test_data) / 5000) * 100
        print(f"Epoch {epoch + 1} complete | Accuracy: {accuracy:.2f}%")
        save_model(weights, biases, epoch, accuracy)

# forward propogation.
def forward(data_input_layer, activations, weights, biases):
    # input layer
    activations[0] = data_input_layer

    # hidden layer 1
    for i in range(16):
        z = sum(weights[0][i][j] * activations[0][j] for j in range(784)) + biases[0][i]  # (z = sigma(wa) + b)
        zs[0][i] = z # update previous z value
        activations[1][i] = relu(z)  # updates activations

    # hidden layer 2
    for i in range(16):
        z = sum(weights[1][i][j] * activations[1][j] for j in range(16)) + biases[1][i]  # (z = sigma(wa) + b)
        zs[1][i] = z  # update previous z value
        activations[2][i] = relu(z)  # updates activations

    # output layer
    for i in range(10):
        z = sum(weights[2][i][j] * activations[2][j] for j in range(16)) + biases[2][i]   # (z = sigma(wa) + b)
        zs[2][i] = z  # update previous z value

    activations[3] = softmax(zs[2])  # update output layer

# algorithm for computing the gradient.
def backpropogation(target):
    global delta_weights, delta_biases

    # reset gradient
    for l in range(3):
        for i in range(len(delta_biases[l])):
            delta_biases[l][i] = 0.0
            for j in range(len(delta_weights[l][i])):
                delta_weights[l][i][j] = 0.0

    # output layer delta
    delta_output = [0.0] * 10
    for i in range(10):
        a = activations[3][i]
        delta_output[i] = a - target[i]
        delta_biases[2][i] = delta_output[i]
        for j in range(16):
            delta_weights[2][i][j] = delta_output[i] * activations[2][j]

    # hidden layer 2 delta
    delta_hidden2 = [0.0] * 16
    for i in range(16):
        error = sum(weights[2][k][i] * delta_output[k] for k in range(10))
        delta_hidden2[i] = error * relu_prime(zs[1][i])
        delta_biases[1][i] = delta_hidden2[i]
        for j in range(16):
            delta_weights[1][i][j] = delta_hidden2[i] * activations[1][j]

    # hidden layer 1 delta
    delta_hidden1 = [0.0] * 16
    for i in range(16):
        error = sum(weights[1][k][i] * delta_hidden2[k] for k in range(16))
        delta_hidden1[i] = error * relu_prime(zs[0][i])
        delta_biases[0][i] = delta_hidden1[i]
        for j in range(784):
            delta_weights[0][i][j] = delta_hidden1[i] * activations[0][j]

# for the output layer.
def softmax(zs):
    exps = [exp(z) for z in zs]
    total = sum(exps)
    return [e / total for e in exps]

# test the network on the test data.
def evaluate(data):
    correct = 0
    for x, label in data:
        forward(x, activations, weights, biases)
        prediction = activations[3].index(max(activations[3]))
        if prediction == label:
            correct += 1
    return correct

# delta_v = -n*grad(C)
def update_weights(learning_rate):
    for l in range(3):
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * delta_weights[l][i][j]
            biases[l][i] -= learning_rate * delta_biases[l][i]

# the output we hoped for.
def target_list(label):
    layer = [0.0] * 10
    layer[label] = 1.0
    return layer

# push the weights and biases in a folder, so that the model can be accessed later.
def save_model(weights, biases, epoch, accuracy):
    filename = f"./models/model_{epoch + 1}_accuracy_{accuracy:.2f}.json"
    with open(filename, "w") as f:
        json.dump({"weights": weights, "biases": biases }, f)

if __name__ == "__main__":
    training_data = load_data("./data/mnist_train.csv")
    test_data = load_data("./data/mnist_test.csv")[:5000]
    gradient_descent(training_data, test_data, epochs = 30, learning_rate = 0.005)
