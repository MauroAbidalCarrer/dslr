import numpy as np
import pickle

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
        # self.weights = np.ones((input_size, nb_outputs))
        self.weights = np.random.randn(input_size, nb_outputs) * 0.1
        # Representing the biases as a column matrix to be able to add then to the matrix product of inputs-weights.
        self.biases = np.zeros((1, nb_outputs))

    def infer(self, inputs):
        """
        inputs:  must be a matrix where each row is an input and each column is a feature.
        returns: a matrix of all the inferred outputs, each row is a one hot guess.
        """
        self.inputs = inputs
        weighted_sums = np.dot(inputs, self.weights)
        self.biased_weighted_sums_outputs = self.biases + weighted_sums
        self.outputs = sigmoid(self.biased_weighted_sums_outputs)
        return self.outputs

    def calculate_loss(self, expected_outputs):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        outputs_clipped = np.clip(self.outputs, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(expected_outputs * np.log(outputs_clipped) +
        (1 - expected_outputs) * np.log(1 - outputs_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses

    def calculate_loss_gradient(self, expected_outputs):
        outputs_len = len(self.outputs[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(self.outputs, 1e-7, 1 - 1e-7)
        # Calculate gradient
        loss_gadients = -(expected_outputs / clipped_dvalues -
        (1 - expected_outputs) / (1 - clipped_dvalues)) / outputs_len
        # Normalize gradient
        loss_gadients = loss_gadients / self.outputs.shape[0]
        return loss_gadients

    # def backward(self, expected_outputs):
    def backpropagation(self, expected_outputs, learning_rate):
        loss_gadients = self.calculate_loss_gradient(expected_outputs)
        weighted_sum_gradients = sigmoid_derivative(self.biased_weighted_sums_outputs) * loss_gadients
        weights_gradients = np.dot(self.inputs.T, weighted_sum_gradients)
        biases_gradients = np.sum(loss_gadients, axis=0, keepdims=True)
        self.biases -= biases_gradients * learning_rate
        self.weights -= weights_gradients * learning_rate

    def calculate_mean_accuracy(self, expected_categorical_outputs):
        # Convert confidance outputs into categorical outputs.
        model_categorical_outputs = np.argmax(self.outputs, axis=1)
        # Make a score array of zeros(incorrect outputs) and ones(correct outputs).
        scores = model_categorical_outputs==expected_categorical_outputs
        return np.mean(scores)


    def train(self, inputs, expected_outputs):
        onehot_expected_outputs = np.eye(4)[expected_outputs]
        
        learning_rate = 0.01
        for _ in range(1500):
            outputs = self.infer(inputs)
            losses = self.calculate_loss(onehot_expected_outputs)
            self.backpropagation(onehot_expected_outputs, learning_rate)
            print('mean loss:', np.mean(losses), '\taccuracy:', self.calculate_mean_accuracy(expected_outputs))

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            weights, biases = pickle.load(file)
        
        # Initialize an instance of the class
        instance = cls(weights.shape[0], weights.shape[1])
        instance.weights = weights
        instance.biases = biases
        return instance