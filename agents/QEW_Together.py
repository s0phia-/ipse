import numpy as np
import random
import time
from scipy.stats import pearsonr
from agents.stew.utils import create_diff_matrix
from utils import random_tiebreak_argmax, fit_lin_reg, fit_ew, fit_ridge, fit_stew


class QTogetherAgent:
    """
    Q-learning-derived algorithm regularised with equal weights
    Learn a mapping from state action pairs to max_a(Q(s',a)) + reward
    Similar to the implementation of DQN, but directly fits the closed form solution to the linear function approximator
    """
    def __init__(self, num_features, actions, regularisation_strength=None, exploration=.15):
        self.experience_window = 1000000000
        self.epsilon = exploration
        self.num_features = num_features
        self.num_actions = actions.n
        self.X = np.zeros([0, self.num_features * self.num_actions])
        self.y = np.zeros([0, 1])
        self.beta = np.random.uniform(low=0, high=1, size=[self.num_actions * self.num_features])
        self.action_space = actions
        self.reward_scale = 1
        self.lam = regularisation_strength
        self.gamma = .95
        self.feature_directions = np.ones[self.num_features * self.num_actions]
        self.feature_direction_switch = self.feature_directions  # convert from old feature directions to new

    def adjust_reg_param(self, action):
        n = self.X.shape[0]
        self.lam = 25**(1/np.exp(n/500))-1

    def learn_feature_directions(self):
        pass
        # corr = []
        # for i in range(self.num_features):
        #     unicorr = pearsonr(self.X.take(i, 1), self.y)
        #     if unicorr >= 0:
        #         corr[i] = 1
        #     else:
        #         corr[i] = -1
        # old_feature_directions = self.feature_directions
        # self.feature_directions = corr
        # self.feature_direction_switch = old_feature_directions * corr
        # self.X = np.matmul(self.X, self.feature_direction_switch)

    def apply_bf(self, state, action):
        sa_bf = np.zeros([self.num_actions, self.num_features])
        sa_bf[action] = state
        return sa_bf.flatten()

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            action = self.get_highest_q_action(state)[0]
        return action

    def store_data(self, state_features, action, reward, state_prime_features):
        """
        Add the latest observation to X and y
        """
        est_return_s_prime = reward * self.reward_scale + (self.gamma *
                                                           self.get_highest_q_action(state_prime_features)[1])
        state_action_features = self.apply_bf(state_features, action)
        self.X = np.vstack([self.X, state_action_features])
        self.y = np.append(self.y, est_return_s_prime)
        if self.X.shape[0] > self.experience_window:
            self.X = np.delete(self.X, 0, axis=0)
            self.y = np.delete(self.y, 0, axis=0)

    def get_highest_q_action(self, state_features):
        """
        Solves argmax_a(Q(s,a)) for given s
        :return: the action with the highest expected return in the given state, and the corresponding expected return
        """
        weights = self.beta.reshape([self.num_actions, self.num_features])
        q_values = np.matmul(weights, state_features)
        argmax_action = random_tiebreak_argmax(q_values)
        return argmax_action, q_values[argmax_action]

    def run(self, env, episodes, max_episode_length=200, sleep_time=0, stopping_criteria=None, *args):
        for i in range(episodes):
            env.reset()
            state = env.state_features
            for _ in range(max_episode_length):
                time.sleep(sleep_time)
                action = self.epsilon_greedy(state)  # env.action_space.sample()
                _, reward, done, info = env.step(action)
                state_prime = info["state_features"]
                self.store_data(state, action, reward, state_prime)
                self.learn_feature_directions()
                #  self.adjust_reg_param(action)  # TODO
                self.learn(state, action, reward, state_prime)
                state = state_prime
                if done or i == max_episode_length - 1:
                    break


###############################################################
# create agents as extensions of Q Agent - for easier looping #
###############################################################

############################
# Agents that fit directly #
############################


class QEwAgent(QTogetherAgent):
    def learn(self, *args):
        self.beta = fit_ew(self.X)


class QStewTogetherAgent(QTogetherAgent):
    def __init__(self, num_features, actions, regularisation_strength, exploration=.15):
        super().__init__(num_features, actions, regularisation_strength, exploration)
        self.D = create_diff_matrix(num_features=self.num_features * self.num_actions)

    def learn(self, *args):
        self.beta = fit_stew(self.X, self.y, self.D, self.lam)


class QRidgeTogetherAgent(QTogetherAgent):
    def learn(self, *args):
        if self.X.shape[1] < 20:
            pass
        else:
            self.beta = fit_ridge(self.X, self.y, self.lam)


class QLinRegTogetherAgent(QTogetherAgent):
    def learn(self, *args):
        if self.X.shape[1] < 100:
            pass
        else:
            self.beta = fit_lin_reg(self.X, self.y)

###################################
# Agents that learn incrementally #
###################################


class QTogInc(QTogetherAgent):
    def __init__(self, num_features, actions, regularisation_strength, exploration=.15):
        super().__init__(num_features, actions, regularisation_strength, exploration)
        self.lr = 0.01
        self.D = create_diff_matrix(num_features=self.num_features * self.num_actions)
        self.matrix_id = np.eye(self.num_actions*self.num_features)

    def get_td_error(self, state, action, state_prime, reward):
        """
        used for incremental updates to weights vector
        """
        a = reward + (self.gamma * self.get_highest_q_action(state_prime)[1])
        b = np.matmul(self.apply_bf(state, action), self.beta)
        return b-a


class QStewTogInc(QTogInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        reg = self.lam * np.matmul(self.D, self.beta)
        delta = self.lr * ((td_err * self.apply_bf(state, action)) + reg)
        self.beta -= delta


class QRidgeTogInc(QTogInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        reg = self.lam * np.matmul(self.matrix_id, self.beta)
        delta = self.lr * ((td_err * self.apply_bf(state, action)) + reg)
        self.beta -= delta


class QLinRegTogInc(QTogInc):
    def learn(self, state, action, reward, state_prime):
        td_err = self.get_td_error(state, action, state_prime, reward)
        delta = self.lr * (td_err * self.apply_bf(state, action))
        self.beta -= delta
