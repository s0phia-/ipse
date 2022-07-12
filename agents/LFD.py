import numpy as np
from scipy.stats import pearsonr
import time


class LFD:
    def __init__(self, num_features, num_actions):
        self.num_features = num_features
        self.num_actions = num_actions
        self.deciding_feature_directions = np.zeros([self.num_features * self.num_actions])
        self.feature_directions = np.ones([self.num_features * self.num_actions])
        self.old_feature_directions = np.ones([self.num_features * self.num_actions])
        self.decided = np.zeros(self.num_features * self.num_actions)
        self.feature_directions_decided = False
        self.weight_memory = []
        self.X = None

    def learn_feature_directions(self):
        if self.X.shape[0] < 2:
            return
        feature_direction_switch = np.zeros(self.num_features*self.num_actions)
        for i in range(self.num_features*self.num_actions):
            unicorr = pearsonr(self.X.take(i, 1), self.y)
            if unicorr[0] < -0 and unicorr[1] < 0.0001:
                feature_direction_switch[i] = -1
            else:
                feature_direction_switch[i] = 1
        self.old_feature_directions = self.feature_directions
        self.feature_directions = feature_direction_switch * self.old_feature_directions
        self.X = np.multiply(self.X, feature_direction_switch)

    def run_lfd_every_step(self, env, episodes, max_episode_length=200, sleep_time=0, stopping_criteria=None, *args):
        for i in range(episodes):
            env.reset()
            state = env.state_features
            for _ in range(max_episode_length):
                time.sleep(sleep_time)
                state, action, reward, state_prime, done = self.step(state, env)
                self.learn(state, action, reward, state_prime)
                self.learn_feature_directions()
                self.weight_memory.append(self.beta)
                state = state_prime
                if done or i == max_episode_length - 1:
                    break

    def run_lfd_once(self, env, episodes, max_episode_length=200, sleep_time=0, stopping_criteria=None, *args):
        """
        run lfd once at start
        """
        for i in range(episodes):
            env.reset()
            state = env.state_features
            for k in range(max_episode_length):
                self.old_feature_directions = self.feature_directions
                # just once, learn feature directions
                if i == 0 and k == 0:
                    self.lfd_once()
                # normal run loop
                time.sleep(sleep_time)
                feature_direction_switch = self.feature_directions * self.old_feature_directions
                self.X = np.multiply(self.X, feature_direction_switch)
                state, action, reward, state_prime, done = self.step(state, env)
                self.learn(state, action, reward, state_prime)
                state = state_prime
                if done or i == max_episode_length - 1:
                    break

    def run_lfd_until_decided(self, env, episodes, max_episode_length=200, sleep_time=0, stopping_criteria=None):
        for i in range(episodes):
            env.reset()
            state = env.state_features
            for k in range(max_episode_length):
                time.sleep(sleep_time)
                if not self.feature_directions_decided:
                    if self.X.shape[0] < 2:
                        _, _, _, state_prime, _ = self.step(state, env)
                        state = state_prime
                    else:
                        self.feature_directions_decided = self.lfd_once()
                        _, _, _, state_prime, _ = self.step(state, env)
                        state = state_prime
                    if self.feature_directions_decided:
                        self.feature_directions = self.deciding_feature_directions
                        self.X = np.multiply(self.X, self.feature_directions)
                        self.old_feature_directions = self.feature_directions
                        print("Feature directions decided at episode " + str(i) + " step " + str(k) + " as " +
                              str(self.feature_directions))
                else:
                    state, action, reward, state_prime, done = self.step(state, env)
                    self.learn(state, action, reward, state_prime)
                    state = state_prime
                    if done or i == max_episode_length - 1:
                        break

    def lfd_once(self, p=0.001):
        for j in range(self.num_features * self.num_actions):
            if self.decided[j] == 1:
                continue
            unicorr = pearsonr(self.X.take(j, 1), self.y)
            if unicorr[1] < p:
                self.decided[j] = 1
                self.deciding_feature_directions[j] = -1 if unicorr[0] < 0 else 1
        if sum(self.decided) == self.decided.shape[0]:
            return True
        else:
            return False

