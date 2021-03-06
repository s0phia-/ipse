import numpy as np
import random
import time
from agents.stew.utils import create_diff_matrix
from utils import random_tiebreak_argmax


class Lstdq:
    """
    Least squares temporal difference Q learning
    """
    def __init__(self, num_features, num_actions, policy, source_of_samples, regularisation_strength=None):
        self.num_features = num_features
        self.num_actions = num_actions
        self.lam = regularisation_strength
        self.gamma = .95  # discount factor
        self.matrix_A = np.zeros([self.num_features*self.num_actions, self.num_features*self.num_actions])
        self.vector_b = np.zeros([self.num_features * self.num_actions])
        self.samples = source_of_samples  # called D in LSPI paper
        self.policy_matrix = policy

    def apply_bf(self, state, action):
        sa_bf = np.zeros([self.num_actions, self.num_features])
        sa_bf[action] = state
        return sa_bf.flatten()

    def greedy_policy(self, state):
        q_values = np.matmul(self.policy_matrix, state)
        return random_tiebreak_argmax(q_values)

    def find_a_and_b(self):
        for d_i in self.samples:
            state, action, reward, state_prime = d_i
            state_action = self.apply_bf(state, action)
            state_action_prime = self.apply_bf(state_prime, self.greedy_policy(state_prime))
            x = state_action - self.gamma * state_action_prime
            self.matrix_A += np.matmul(state_action.reshape([state_action.shape[0], 1]), x.reshape([1, x.shape[0]]))
            self.vector_b += state_action * reward

    def fit(self):
        self.find_a_and_b()
        policy = np.matmul(np.linalg.inv(self.matrix_A), self.vector_b)
        return policy.reshape([self.num_actions, self.num_features])


class LstdqEw(Lstdq):
    """
    Least squares temporal difference Q learning, regularised with equal weights
    """
    def __init__(self, num_features, num_actions, policy, source_of_samples, regularisation_strength):
        super().__init__(num_features, num_actions, policy, source_of_samples, regularisation_strength)
        self.matrix_DtD = create_diff_matrix(num_features*num_actions)  # ew regularisation matrix DtD

    def fit(self):
        self.find_a_and_b()
        policy = np.matmul(np.linalg.inv(self.matrix_A + self.lam*self.matrix_DtD), self.vector_b)
        return policy.reshape([self.num_actions, self.num_features])


class LstdqL2(Lstdq):
    """
    Least squares temporal difference Q learning, regularised with equal weights
    """
    def __init__(self, num_features, num_actions, policy, source_of_samples, regularisation_strength):
        super().__init__(num_features, num_actions, policy, source_of_samples, regularisation_strength)
        self.matrix_id = np.eye(num_features * num_actions)

    def fit(self):
        self.find_a_and_b()
        policy = np.matmul(np.linalg.inv(self.matrix_A + self.lam*self.matrix_id), self.vector_b)
        return policy.reshape([self.num_actions, self.num_features])


class LspiAgent:
    def __init__(self, num_features, actions, regularisation_strength, max_samples=10**6, source_of_samples=[]):
        self.source_of_samples = source_of_samples
        self.max_samples = max_samples
        self.num_features = num_features
        self.num_actions = actions.n
        self.reg_strength = regularisation_strength
        self.policy = np.zeros([self.num_actions, self.num_features])
        self.epsilon = 1 # TODO
        self.action_space = actions
        self.model = Lstdq

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            q_values = np.matmul(self.policy, state)
            action = random_tiebreak_argmax(q_values)
        return action

    def learn(self, stopping_criteria, max_out=100000):
        diff = stopping_criteria + 1
        i = 1
        for i in range(max_out):
            if diff > stopping_criteria:
                w_old = self.policy
                agent = self.model(self.num_features, self.num_actions, w_old, self.source_of_samples,
                                   self.reg_strength)
                w_new = agent.fit()
                self.policy = w_new
                diff = np.linalg.norm(w_old-w_new)
                i += 1
            else:
                break
        return w_new

    def collect_experience(self, episodes, env, max_episode_length, sleep_time):
        new_samples = []
        for i in range(episodes):
            env.reset()
            state = env.state_features
            for _ in range(max_episode_length):
                time.sleep(sleep_time)
                action = self.epsilon_greedy(state)
                _, reward, done, info = env.step(action)
                state_prime = info["state_features"]
                new_samples.append([state, action, reward, state_prime])
                state = state_prime
                if done or len(new_samples) >= self.max_samples:
                    break
        self.source_of_samples += new_samples
        if len(self.source_of_samples) >= self.max_samples:  # if too many samples, only keep last N
            self.source_of_samples = self.source_of_samples[-self.max_samples:]

    def run(self, episodes, env, stopping_criteria, max_episode_length=200, sleep_time=0):
        self.collect_experience(episodes=episodes, env=env, max_episode_length=max_episode_length,
                                sleep_time=sleep_time)
        self.learn(stopping_criteria=stopping_criteria)


class LspiAgentEw(LspiAgent):
    def __init__(self, num_features, actions, regularisation_strength, max_samples=10**5, source_of_samples=[]):
        super().__init__(num_features, actions, regularisation_strength, max_samples, source_of_samples)
        self.model = LstdqEw


class LspiAgentL2(LspiAgent):
    def __init__(self, num_features, actions, regularisation_strength, max_samples=10**5, source_of_samples=[]):
        super().__init__(num_features, actions, regularisation_strength, max_samples, source_of_samples)
        self.model = LstdqL2
