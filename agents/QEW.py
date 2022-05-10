import numpy as np
import random
import time
from agents.stew.utils import create_diff_matrix
from utils import random_tiebreak_argmax, fit_lin_reg, fit_ew, fit_ridge, fit_stew


class QAgent:
    """
    Q-learning-derived algorithm regularised with equal weights
    Learn a mapping from state action pairs to max_a(Q(s',a)) + reward
    Similar to the implementation of DQN, but directly fits the closed form solution to the linear function approximator
    """
    def __init__(self, num_features, actions, regularisation_strength=None, exploration=.15):
        self.experience_window = 10000
        self.epsilon = exploration
        self.num_features = num_features
        self.num_actions = actions.n
        self.X = np.zeros([0, self.num_features * self.num_actions])
        self.y = np.zeros([0, 1])
        self.beta = np.random.uniform(low=0, high=1, size=[self.num_actions * self.num_features])
        self.action_space = actions
        self.reward_scale = 1/100
        self.lam = regularisation_strength

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
        est_return_s_prime = self.get_highest_q_action(state_prime_features)[1] + reward*self.reward_scale
        state_action_features = np.zeros([self.num_actions, self.num_features])
        state_action_features[action] = state_features
        self.X = np.vstack([self.X, state_action_features.flatten()])
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

        all_state_actions = []
        for i in range(0, self.num_actions):
            z = np.zeros([self.num_actions, self.num_features])
            z[i] = state_features
            all_state_actions.append(z.flatten())
        all_state_action_q_values = [np.matmul(self.beta.transpose(), x) for x in all_state_actions]

        assert all_state_action_q_values == q_values
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
                self.learn(state, action, reward, state_prime)
                state = state_prime
                if done or i == max_episode_length - 1:
                    break


###############################################################
# create agents as extensions of Q Agent - for easier looping #
###############################################################

class QStewAgentType1(QAgent):
    def __init__(self, num_features, actions, regularisation_strength, exploration=.15):
        super().__init__(num_features, actions, regularisation_strength, exploration)
        self.D = create_diff_matrix(num_features=self.num_features * self.num_actions)

    def learn(self):
        self.beta = fit_stew(self.X, self.y, self.D, self.lam)


class QRidgeAgentType1(QAgent):
    def learn(self):
        self.beta = fit_ridge(self.X, self.y, self.lam)


class QEwAgentType1(QAgent):
    def learn(self):
        self.beta = fit_ew(self.X)


class QLinRegType1(QAgent):
    def learn(self):
        self.beta = fit_lin_reg(self.X, self.y)
