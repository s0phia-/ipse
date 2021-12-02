"""
Environment based on the MDP described in David Silver's RL course
Slide 25, link: https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""

import numpy as np


class StudentDilemma:

    def __init__(self):
        # states:
        # 0: on social media
        # 1: nothing
        # 2: study 1
        # 3: study 2
        # 4: sleep and fail, end state
        # 5: study 3 and pass, end state
        # 6: pub, random state
        self.state_space = [0, 1, 2, 3, 4, 5, 6]
        self.terminal_states = {4, 5}
        # fun | fulfillment | tiredness | parent disappointment
        self.state_features = [[0.2, -1, 0.1, 1],
                               [0, 0, 0, 0],
                               [-1, 0.1, 0.5, -0.2],
                               [-1, 0.3, 0.7, -0.2],
                               [0, 0, -1, 1],
                               [-1, 1, 0.9, -1],
                               [1, 0.2, 0.4, 1]]
        self.rewards = [-1, 0, -2, -2, 0, 10, 1]
        self.deterministic_states = {0, 1, 2, 3, 4, 5}
        self.nondeterministic_states = {6}
        err_msg = "Each state must be classed as deterministic or nondeterministic"
        assert set.union(self.deterministic_states, self.nondeterministic_states) == set(self.state_space), err_msg
        # deterministic states map actions to resulting states
        # nondeterministic states map to states with transition probabilities
        # actions:
        # 0: study
        # 1: pub
        # 2: sleep
        # 3: social media
        # 4: quit social media
        self.transitions = [[None, None, None, 0, 1],
                            [2, None, None, 0, None],
                            [3, None, 4, None, None],
                            [5, 6, None, None, None],
                            [None, None, None, None, None],
                            [None, None, None, None, None],
                            [0, .2, .4, .4, 0, 0, 0]]
        self.state = 1
        self.action_space = self.get_available_actions()
        self.done = False

    def get_available_actions(self):
        return JankyActionSpace([i for i, v in enumerate(self.transitions[self.state]) if v is not None])

    def step(self, action):
        err_msg = f"{action} invalid in this state."
        assert action in self.get_available_actions(), err_msg
        # perform action to move to next state
        next_state = self.transitions[self.state][action]
        self.state = next_state
        reward = self.rewards[self.state]
        # if next state is random, don't give back control to user until out of a random state
        while next_state in self.nondeterministic_states:
            # so the user can see which random states are being visited
            self.render()
            next_state = np.random.choice(self.state_space, p=self.transitions[self.state])
            self.state = next_state
            # return cumulative reward after the agent is out of a random state
            reward += self.rewards[self.state]
        self.action_space = self.get_available_actions()
        if self.state in self.terminal_states:
            self.done = True
        return np.array(self.state_features[self.state]), reward, self.done, {}

    def reset(self):
        self.done = False
        self.state = 1

    def render(self):
        print(f"Current state: {self.state}")

    def close(self):
        pass

class JankyActionSpace(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def sample(self):
        if self.size > 0:
            return np.random.choice(self)
        else:
            return None
