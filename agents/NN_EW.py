import numpy as np
import random
import math
from agents.stew.utils import create_diff_matrix


class EwDefaultDict(dict):

    def __missing__(self, key):
        return create_diff_matrix(key)


class NeuralNetwork:
    def __init__(self, x_size, y_size):
        self.x = x  # input
        self.y = y  # target
        self.weights_layer_1 = np.random.rand(self.x.shape[1], 12)  # weights input to hidden_1
        self.hidden_1 = None  # hidden layer 1
        self.weights_layer_2 = np.random.rand(12, self.y.shape)  # weights hidden_1 to output
        self.y_hat = np.zeros(self.y.shape)  # output layer

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return x * (1 - x)

    def feedforward(self, activation):
        self.hidden_1 = activation(np.dot(self.x, self.weights_layer_1))
        self.y_hat = activation(np.dot(self.hidden_1, self.weights_layer_2))

    @staticmethod
    def derivative_ew(x):
        return np.dot(EwDefaultDict[x.shape[0]], x)

    def backprop(self, d_activation):
        """
        loss function: 1/2||y-y_hat||^2_2 + lam*D*w
        where D the equal weight regularisation matrix
        Regularises across layers
        """
        err = self.y - self.y_hat
        z1 = np.dot(self.weights_layer_1, self.x)
        z2 = np.dot(self.weights_layer_2, self.hidden_1)

        d_weights_layer_2 = np.dot(self.hidden_1.transpose, err * d_activation(z2)) +\
                            self.derivative_ew(self.weights_layer_2)
        d_weights_layer_1 = np.dot(self.x.transpose,
                                   np.dot(err * d_activation(z2), self.weights_layer_2.transpose)
                                   * d_activation(z1))
        self.weights_layer_1 += d_weights_layer_1
        self.weights_layer_2 += d_weights_layer_2


class DqnEwAgent:
    """
    A DQN style agent with equal weights regularisation.
    """

    def __init__(self, num_features, actions, regularisation_strength, exploration=.15):
        self.num_actions = actions.n
        self.num_features = num_features
        self.lam = regularisation_strength
        self.epsilon = exploration
        self.actions = actions
        self.learning_rate = 0.01
        self.gamma = 0.9  # discount factor
        self.batch_size = 32
        self.buffer_size = 1e5
        self.nn = NeuralNetwork()

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.actions.sample()
        else:
            action = self.get_highest_q_action(state)[0]
        return action

    def learn(self):
        pass

    def run(self):
        pass

    def get_highest_q_action(self, state):
        pass
