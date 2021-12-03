from gym.envs.classic_control import CartPoleEnv
import math
import numpy as np


class CartPoleRBF(CartPoleEnv):
    def __init__(self):
        super().__init__()
        # setup features based on Least-Squares Policy Iteration
        self.theta_rbf = [-math.pi / 4, 0, math.pi / 4]
        self.omega_rbf = [-1, 0, 1]
        self.features = np.zeros((2, 10))

    def get_features(self, action):
        theta = self.state[2]
        omega = self.state[3]
        rbf = [math.exp(-np.linalg.norm([theta - self.theta_rbf[i], omega - self.omega_rbf[j]]) / 2)
               for i in range(len(self.theta_rbf)) for j in range(len(self.omega_rbf))]
        features = np.zeros(self.features.shape)
        features[action, :] = np.append([1], rbf)
        self.features = features

    def step(self, action):
        _, reward, done, info = super().step(action)
        self.get_features(action)
        return self.features, reward, done, info

    def lookahead_one_step(self):
        current_state = self.state
        current_action_space = self.action_space
        current_steps_beyond_done = self.steps_beyond_done
        current_features = self.features
        next_state_actions = []
        for a in self.discrete_to_list():
            next_state_actions.append(self.step(a))
            self.state = current_state
            self.action_space = current_action_space
            self.steps_beyond_done = current_steps_beyond_done
            self.features = current_features
        return next_state_actions

    def discrete_to_list(self, start=0):
        """Convert a discrete object as defined in AI gym to a list"""
        end = self.action_space.n
        return list(range(start, end))