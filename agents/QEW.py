import numpy as np
import random
from agents.stew.utils import create_diff_matrix
from utils import random_tiebreak_argmax, fit_lin_reg, fit_ew, fit_ridge, fit_stew


class QEW:
    """
    Q-learning-derived algorithm regularised with equal weights
    Learn a mapping from state action pairs to max_a(Q(s',a)) + reward
    Similar to the implementation of DQN, but directly fits the closed form solution to the linear function approximator
    """
    def __init__(self, num_features, actions, regularisation_strength=0.1, exploration=.15, model="stew"):
        self.experience_window = 1000
        self.lam = regularisation_strength
        self.epsilon = exploration
        self.num_features = num_features
        self.num_actions = actions.n
        self.num_action_state_features = num_features * self.num_actions
        self.D = create_diff_matrix(num_features=self.num_action_state_features)
        self.X = np.zeros([0, self.num_action_state_features])
        self.y = np.zeros([0, 1])
        self.beta = np.random.uniform(low=0, high=1, size=self.num_action_state_features)
        self.action_space = actions
        self.model = model
        self.reward_scale = 1/100

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            action = self.get_highest_q_action(state)[0]
        return action

    def append_data(self, state_features, action,  reward, state_prime_features):
        """
        Add the latest observation to X and y
        """
        est_return_s_prime = self.get_highest_q_action(state_prime_features)[1] + reward*self.reward_scale
        state_action_features = np.zeros([self.num_actions, self.num_features])
        state_action_features[action] = state_features
        self.X = np.vstack([self.X, state_action_features.flatten()])
        self.y = np.append(self.y, est_return_s_prime)
        if self.X.shape[0] > self.experience_window:
            self.X = np.delete(self.X, 0, axis=0)
            self.y = np.delete(self.y, 0, axis=0)
        err_msg = "Shape of X and y do not match"
        assert self.X.shape[0] == self.y.shape[0], err_msg
        err_msg = "Too many data points"
        assert self.X.shape[0] <= self.experience_window, err_msg

    def learn(self, state_features, action, reward, state_prime_features):
        self.append_data(state_features, action, reward, state_prime_features)
        if self.model == "stew":
            self.beta = fit_stew(self.X, self.y, self.D, self.lam)
        elif self.model == "ew":
            self.beta = fit_ew(self.X)
        elif self.model == "ridge":
            self.beta = fit_ridge(self.X, self.y, self.lam)
        elif self.model == "lin_reg":
            self.beta = fit_lin_reg(self.X, self.y)
        else:
            raise ValueError('Please choose a valid model name {stew, ew, or ridge}.')

    def get_highest_q_action(self, state_features):
        """
        Solves argmax_a(Q(s,a)) for given s
        :return: the action with the highest expected return in the given state, and the corresponding expected return
        """
        all_state_actions = []
        for i in range(0, self.num_actions):
            z = np.zeros([self.num_actions, self.num_features])
            z[i] = state_features
            all_state_actions.append(z.flatten())
        all_state_action_q_values = [np.matmul(self.beta.transpose(), x) for x in all_state_actions]
        argmax_action = random_tiebreak_argmax(all_state_action_q_values)
        return argmax_action, all_state_action_q_values[argmax_action]
