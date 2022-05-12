import numpy as np
from collections import defaultdict
from agents.QEW_Together import QTogetherAgent
from agents.stew.utils import create_diff_matrix
from utils import random_tiebreak_argmax, fit_lin_reg, fit_ew, fit_ridge, fit_stew


class QSeparatedAgent(QTogetherAgent):
    """
    Similar to QEW but state action vector are separated out. This will prevent regularisation from pushing actions
    to similar values
    """
    def __init__(self, num_features, actions, regularisation_strength=None, exploration=.15, *args):
        super().__init__(num_features, actions, regularisation_strength, exploration)
        self.beta = np.random.uniform(low=0, high=1, size=[self.num_actions, self.num_features])
        self.X = defaultdict(lambda: [])
        self.y = defaultdict(lambda: [])
        self.D = create_diff_matrix(num_features=self.num_features)

    def store_data(self, state_features, action, reward, state_prime_features):
        """
        Add the latest observation to X and y
        """
        est_return_s_prime = self.get_highest_q_action(state_prime_features)[1] + (reward*self.reward_scale)
        self.X[action].append(state_features)
        self.y[action].append(est_return_s_prime)
        if len(self.X[action]) > self.experience_window:
            self.X[action].pop(0)
            self.y[action].pop(0)

    def get_highest_q_action(self, state_features):
        """
        Solves argmax_a(Q(s,a)) for given s
        :return: the action with the highest expected return in the given state, and the corresponding expected return
        """
        all_state_action_q_values = [np.matmul(b, state_features) for b in self.beta]
        argmax_action = random_tiebreak_argmax(all_state_action_q_values)
        return argmax_action, all_state_action_q_values[argmax_action]


####################################################################
# create agents as extensions of QEWv2 - allows for easier looping #
####################################################################

############################
# Agents that fit directly #
############################


class QStewSeparatedAgent(QSeparatedAgent):
    def learn(self, state_features, action, reward, state_prime_features):
        self.beta[action] = fit_stew(self.X[action], self.y[action], self.D, self.lam)


class QRidgeSeparatedAgent(QSeparatedAgent):
    def learn(self, state_features, action, reward, state_prime_features):
        self.beta[action] = fit_ridge(self.X[action], self.y[action], self.lam)


class QLinRegSeparatedAgent(QSeparatedAgent):
    def learn(self, state_features, action, reward, state_prime_features):
        if len(self.y[action]) < 1000:
            pass
        else:
            self.beta[action] = fit_lin_reg(self.X[action], self.y[action])


###################################
# Agents that learn incrementally #
###################################

class QSepInc(QSeparatedAgent):
    def __init__(self, num_features, actions, regularisation_strength, exploration=.15):
        super().__init__(num_features, actions, regularisation_strength, exploration)
        self.lr = 0.001
        self.gamma = 0.9
        self.matrix_id = np.eye(self.num_features)

    def get_td_error(self, state, action, state_prime, reward):
        """
        used for incremental updates to weights vector
        """
        a = reward + self.gamma * self.get_highest_q_action(state_prime)[1]
        b = np.matmul(state, self.beta[action])
        return a-b


class QStewSepInc(QSepInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        reg = self.lam * np.matmul(self.D, self.beta[action])
        delta = self.lr * ((td_err * state) + reg)
        self.beta[action] += delta


class QRidgeSepInc(QSepInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        reg = self.lam * np.matmul(self.matrix_id, self.beta[action])
        delta = self.lr * ((td_err * state) + reg)
        self.beta[action] += delta


class QLinRegSepInc(QSepInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        delta = self.lr * td_err * state
        self.beta[action] += delta

