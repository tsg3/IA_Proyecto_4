import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.random.randint(2, size = 10)
        self.count = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.bias_1 = 2
        self.bias_2 = 0
        self.learning_rate = learning_rate

    def activation(self, x): # Sigmoid
        return 1 / (1 + np.exp(-x + 50))

    def d_activation(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def guess(self, input):
        layer_1 = self.bias_1 * np.dot(self.count, np.dot(input, self.weights)) \
            + self.bias_2 # layer 1
        return self.activation(layer_1) # layer 2

    def compute_gradients(self, input, target):
        layer_1_temp = np.dot(self.count, np.dot(input, self.weights))
        layer_1 = self.bias_1 * layer_1_temp + self.bias_2
        layer_2 = self.activation(layer_1) # guess

        d_error_d_layer_2 = 2 * (layer_2 - target)
        d_prediction_d_layer_1 = self.d_activation(layer_1)
        d_layer_1_d_bias_1 = layer_1_temp
        d_layer_1_d_bias_2 = 1
        d_layer_1_d_weights = self.bias_1 * np.dot(self.count, input)

        d_error_d_bias_1 = d_error_d_layer_2 * d_prediction_d_layer_1 * d_layer_1_d_bias_1
        d_error_d_bias_2 = d_error_d_layer_2 * d_prediction_d_layer_1 * d_layer_1_d_bias_2
        d_error_d_weights = d_error_d_layer_2 * d_prediction_d_layer_1 * d_layer_1_d_weights

        return d_error_d_bias_1, d_error_d_bias_2, d_error_d_weights

    def back_propagation(self, d_error_d_bias_1, d_error_d_bias_2, d_error_d_weights): # Generalized Delta Rule
        self.bias_1 = self.bias_1 - d_error_d_bias_1 * self.learning_rate
        self.bias_2 = self.bias_2 - d_error_d_bias_2 * self.learning_rate
        self.weights = self.weights - d_error_d_weights * self.learning_rate

    def train(self, training_vectors, training_targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):

            # index = np.random.randint(len(training_vectors))
            index = current_iteration
            vector = training_vectors[index]
            target = training_targets[index]

            d_error_d_bias_1, d_error_d_bias_2, d_error_d_weights = self.compute_gradients(vector, target)
            self.back_propagation(d_error_d_bias_1, d_error_d_bias_2, d_error_d_weights)

            # if current_iteration % 100 == 0:
            #     cumulative_error = 0
            #     for index in range(100):
            #         vector = training_vectors[index]
            #         target = training_targets[index]
            #         guess = self.guess(vector)
            #         error = np.square(guess - target)
            #         cumulative_error = cumulative_error + error
            #     cumulative_errors.append(cumulative_error)

        return cumulative_errors

def generate_training_data():
    vectors_list = []
    targets_list = []
    oper_a = np.array([[1,], [1,], [1,], [1,], [1,], [1,], [1,], [1,], [1,], [1,]])
    oper_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    for i in range(7643):
        vector = np.random.binomial(1, ((i % 9) + 1) / 10, size = (10, 10))
        target = 1 if np.dot(oper_b, np.dot(vector, oper_a)) > 50 else 0
        vectors_list.append(vector)
        targets_list.append(target)

    return vectors_list, targets_list

if "__main__":
    learning_rate = 0.001
    neural_network = NeuralNetwork(learning_rate)

    vectors_list, targets_list = generate_training_data()
    # print("<<< ------------------------------------------------- >>>")
    # for i in range(len(vectors_list)):
    #     print(f"Dataset {i}")
    #     print("Vector =")
    #     print(vectors_list[i])
    #     print(f"Target = {targets_list[i]}")
    # print("<<< ------------------------------------------------- >>>")

    training_error = neural_network.train(vectors_list, targets_list, 7643)

    # plt.plot(training_error)
    # plt.xlabel("Iterations")
    # plt.ylabel("Error for all training instances")
    # plt.savefig("cumulative_error_main.png")

    # zero_vector = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # )
    # one_vector = np.array(
    #     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # )
    # vector = np.array(
    #     [[1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]]
    # )

    # print(neural_network.guess(vector))
    # print(neural_network.activation(neural_network.bias_1 * np.dot(neural_network.count, np.dot(vector, neural_network.weights)) + neural_network.bias_2))

    print(neural_network.weights)
    print(neural_network.bias_1)
    print(neural_network.bias_2)