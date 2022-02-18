import numpy as np
import random


class QEW:
    def __init__(self, num_features, actions):
        self.experience_window = 500
        self.lam = 0.001  # regularisation strength
        self.epsilon = 0.1  # exploration
        self.enough_data = 20  # choose actions at random until have enough data points
        self.num_features = num_features
        self.num_action_state_features = num_features * actions.n
        self.D = self.create_diff_matrix(num_features=self.num_action_state_features)
        self.X = np.empty([0, self.num_action_state_features])
        self.y = np.empty([0, 1])
        self.beta = np.empty(self.num_action_state_features)
        self.action_space = actions

    def fit_closed_form(self):
        a = np.matmul(self.X.transpose(), self.X) + self.lam * np.matmul(self.D.transpose(), self.D)
        b = np.matmul(self.X.transpose(), self.y)
        self.beta = np.matmul(np.linalg.inv(a), b)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon or self.X.shape[0] < self.enough_data:
            action = self.action_space.sample()
        else:
            action = self.get_highest_q_action(state)[0]
        return action

    def append_data(self, state_features, action,  reward, state_prime_features):
        err_msg = "Shape of X and y do not match"
        assert self.X.shape[0] == self.y.shape[0], err_msg
        est_return_s_prime = self.get_highest_q_action(state_prime_features)[1] + reward
        state_action_features = np.zeros([self.action_space.n, self.num_features])
        state_action_features[action] = state_features
        self.X = np.vstack([self.X, state_action_features.flatten()])
        self.y = np.append(self.y, est_return_s_prime)
        if self.X.shape[0] > self.experience_window:
            self.X = np.delete(self.X, 0, axis=0)
            self.y = np.delete(self.y, 0, axis=0)

    def learn(self, state_features, action, reward, state_prime_features):
        self.append_data(state_features, action, reward, state_prime_features)
        self.fit_closed_form()

    def get_highest_q_action(self, state_features):
        all_state_actions = []
        for i in range(0, self.action_space.n):
            z = np.zeros([self.action_space.n, self.num_features])
            z[i] = state_features
            all_state_actions.append(z.flatten())
        all_state_action_q_values = [np.matmul(self.beta.transpose(), x) for x in all_state_actions]
        argmax_action = np.argmax(all_state_action_q_values)
        return argmax_action, all_state_action_q_values[argmax_action]

    @staticmethod
    def create_diff_matrix(num_features):
        # had to copy function from stew.utils due to Numba import errors
        d = np.full((num_features, num_features), fill_value=-1.0, dtype=np.float_)
        for i in range(num_features):
            d[i, i] = num_features - 1.0
        return d
