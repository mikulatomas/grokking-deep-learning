# simple neural network with 6 inputs and one output
# starts with random weights
# uses gradient descent to converge to the final weights

import numpy as np

# vector as a input
input_data = np.array((0.2, 0.5, 0.1, 0.9, 0.2, 0.1))

# expected output
expected_data = 0.9

# how much to decrease weights update
alpha = 0.1

# init random weights
init_weights = np.random.rand(1, 6)

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

        # delta between prediction and expected result
        delta = result - expected_data

        # mean square error
        error = delta ** 2
        print("Error {}".format(error))

        # update weights via gradient descent
        weights = weights - alpha * (delta * input_data)

        print("New weights {}\n".format(weights))

    return weights


weights = train(input_data,
                expected_data,
                init_weights,
                alpha,
                1000)


# predict with trained weights
print(predict(input_data, weights))
