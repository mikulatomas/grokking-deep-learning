# simple neural network with one input and one output
# starts with random weight
# uses gradient descent to converge to the final weight

import numpy as np

# single number as input
input_data = 2

# expected number as output
expected_data = 0.9

# how much to decrease weights update
alpha = 0.1

# initial random weight (we need only one weight)
init_weights = np.random.uniform()

# predict output


def predict(input_data, weights):
    return input_data * weights

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

        print("New weights\n{}\n".format(weights))

    return weights


weights = train(input_data,
                expected_data,
                init_weights,
                alpha,
                100)

# predict with trained weights
print(predict(input_data, weights))
