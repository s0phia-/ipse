from gym.envs.classic_control import CartPoleEnv
import math
import numpy as np


class CartPoleRBF(CartPoleEnv):
    def __init__(self):
        super().__init__()
        # setup features based on Least-Squares Policy Iteration
        self.theta_rbf = [-math.pi / 4, 0, math.pi / 4]
        self.omega_rbf = [-1, 0, 1]
        self.state_features = np.zeros(10)

    def get_features(self):
        theta = self.state[2]
        omega = self.state[3]
        rbf = [math.exp(-np.linalg.norm([theta - self.theta_rbf[i], omega - self.omega_rbf[j]]) / 2)
               for i in range(len(self.theta_rbf)) for j in range(len(self.omega_rbf))]
        self.state_features = rbf
        return rbf

    def step(self, action):
        _, reward, done, info = super().step(action)
        self.get_features()
        return self.state_features, reward, done, info

    def get_sa_pairs(self):
        current_state = self.state
        current_steps_beyond_done = self.steps_beyond_done
        next_state_actions = []
        for a in self.discrete_to_list():
            next_state_actions.append(self.step(a))
            self.set_to_state(current_state, current_steps_beyond_done)
        return next_state_actions

    def get_current_state(self):
        """return the info necessary to reset the environment to the present state"""
        return self.state, self.steps_beyond_done

    def set_to_state(self, state, steps_beyond_done):
        """reset the environment to a specified state"""
        self.state = state
        self.steps_beyond_done = steps_beyond_done
        self.state_features = self.get_features()

    def discrete_to_list(self, start=0):
        """Convert a discrete object as defined in AI gym to a list"""
        end = self.action_space.n
        return list(range(start, end))
