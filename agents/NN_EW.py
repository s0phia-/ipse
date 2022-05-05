import numpy as np
import random


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


    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.actions.sample()
        else:
            action = self.get_highest_q_action(state)[0]
        return action

    def learn(self):
        pass

    def get_highest_q_action(self, state):
        pass
