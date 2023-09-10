import numpy as np

# Implementing one VS all logistic regression model as a fully connected perceptron layer with sigmoid as activation function.
# Internet article/tuorials are making it look complicated but really that's all it is.

def sigmoid(x):
    return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class Log_regs:
        
    def __init__(self, input_size, nb_outputs):
        # Representing the weights a matrix transposed from the column matrix of the weights for performance.
        self.weights = np.random.rand(input_size, nb_outputs)
        # Representing the biases as a column matrix to be able to add then to the matrix product of inputs-weights.
        self.biases = np.zeros((1, nb_outputs))

    def infer(self, inputs):
        """
        inputs:  must be a matrix where each row is an input and each column is a feature.
        returns: a matrix of all the inferred outputs, each row is a one hot guess.
        """
        self.inputs = inputs
        weighted_sums = np.dot(inputs, self.weights)
        self.biased_weighted_sums = self.biases + weighted_sums
        self.outputs = sigmoid(self.biased_weighted_sums)
        return self.outputs

    def backpropagation(self, gradients, learning_rate):
        weighted_sum_gradients = sigmoid_derivative(self.biased_weighted_sums) * gradients
        print("weighted_sum_gradients:\n", weighted_sum_gradients)
        print("self.inputs.T:\n", self.inputs.T)
        print("self.inputs.T.shape:\n", self.inputs.T.shape)
        print("weighted_sum_gradients:\n", weighted_sum_gradients.shape)
        # weights_gradients = np.dot(self.inputs.T, weighted_sum_gradients)
        weights_gradients = np.dot(self.inputs.T, weighted_sum_gradients)
        print("weights_gradients:\n", weights_gradients)
        biases_gradients = np.sum(gradients, axis=0, keepdims=True)
        print("biases_gradients:\n", biases_gradients)
        self.biases -= biases_gradients * learning_rate
        self.weights -= weights_gradients * learning_rate

    def train(self, inputs, expected_outputs):
        onehot_expected_outputs = np.eye(4)[expected_outputs]
        
        for _ in range(1):
            outputs = self.infer(inputs)
            gradients = np.where(onehot_expected_outputs < outputs, 1, np.where(onehot_expected_outputs > outputs, -1, 0))
            print("gradients:\n", gradients)
            self.backpropagation(gradients, 0.1)
            print('mean accuracy:', np.mean(np.where(outputs == onehot_expected_outputs, 1, 0)))

        # print("weights")
        # print(self.weights)
        # print(self.biases)