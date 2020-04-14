import numpy as np

input_data = np.array((0.2, 0.5, 0.1, 0.9, 0.2, 0.1))
expected_data = 0.9
alpha = 0.1

# init random weights
init_weights = np.random.rand(1, 6)

print(init_weights)


def predict(input_data, weights):
    return weights.dot(input_data)


# random guess
print(predict(input_data, init_weights))


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

        print("New weights {}\n".format(weights))

    return weights


weights = train(input_data,
                expected_data,
                init_weights,
                alpha,
                1000)

print(predict(input_data, weights))
