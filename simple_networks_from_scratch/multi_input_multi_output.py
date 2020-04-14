# simple neural network with 6 inputs and 2 outputs
# starts with random weights
# uses gradient descent to converge to the final weights

import numpy as np

# vector as a input
input_data = np.array((0.2, 0.5, 0.1, 0.9, 0.2, 0.1))

# expected output -> vector
expected_data = np.array((0.9, 0.5))

# how much to decrease weights update
alpha = 0.1

# init random weights
init_weights = np.random.rand(2, 6)

print(init_weights)

# predict output


def predict(input_data, weights):
    return weights.dot(input_data)


# random guess
print(predict(input_data, init_weights))

# train network's weights


def train(input_data, expected_data, init_weights, alpha, number_of_epoch):
    weights = init_weights

    for i in range(number_of_epoch):
        print("Epoch {}".format(i))

        # get prediction
        result = predict(input_data, weights)
        print("Predicted {}".format(result))

        # delta between prediction and expected result -> vector
        delta = result - expected_data
        print("Difference {}".format(delta))

        # mean square error -> vector
        error = delta ** 2
        print("Error {}".format(error))

        # transpose delta matrix before multiplication
        delta_matrix = np.matrix(delta).transpose()
        # cast input to matrix (numpy stuff)
        input_matrix = np.matrix(input_data)

        # update weights via gradient descent
        weights = np.array(
            weights - alpha * delta_matrix.dot(input_matrix))

        print("New weights\n{}\n".format(weights))

    return weights


weights = train(input_data,
                expected_data,
                init_weights,
                alpha,
                1000)

# predict with trained weights
print(predict(input_data, weights))
