import numpy as np

input_data = 2
expected_data = 0.9
alpha = 0.1

init_weights = 0.1


def predict(input_data, weights):
    return input_data * weights


def train(input_data, expected_data, init_weights, alpha, number_of_epoch):
    weights = init_weights

    for i in range(number_of_epoch):
        print("Epoch {}".format(i))

        result = predict(input_data, weights)
        print("Predicted {}".format(result))

        difference = result - expected_data

        error = difference ** 2
        print("Error {}".format(error))

        weights = weights - alpha * (difference * input_data)

        print()

    return weights


weights = train(input_data,
                expected_data,
                init_weights,
                alpha,
                100)

print(predict(input_data, weights))
