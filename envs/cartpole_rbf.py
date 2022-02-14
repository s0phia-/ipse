from gym.envs.classic_control import CartPoleEnv
import numpy as np
import warnings


class CartPoleRBF(CartPoleEnv):
    def __init__(self, informative_rewards=False):
        super().__init__()
        # setup features based on Least-Squares Policy Iteration
        self.theta_rbf = [-np.pi / 4, 0, np.pi / 4]
        self.omega_rbf = [-1, 0, 1]
        # self.theta_rbf = [-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4]
        # self.omega_rbf = [-1, -0.5, 0, 0.5, 1]
        self.rbf_grid = np.array([np.tile(self.theta_rbf, len(self.omega_rbf)),
                                  np.repeat(self.omega_rbf, len(self.theta_rbf))])
        self.state_features = np.zeros(int(len(self.theta_rbf) * len(self.omega_rbf)))
        self.num_features = len(self.state_features)
        self.num_actions = self.action_space.n
        self.informative_rewards = informative_rewards
        if self.informative_rewards:
            warnings.warn("INFO: Using informative rewards. "
                          "Non-original formulation of Cartpole!")

    def get_features(self):
        theta = self.state[2]
        omega = self.state[3]
        # rbf = [np.exp(-np.linalg.norm([theta - self.theta_rbf[i], omega - self.omega_rbf[j]]) / 2)
        #        for i in range(len(self.theta_rbf)) for j in range(len(self.omega_rbf))]
        s = np.array([[theta],
                      [omega]])
        rbf = np.exp(-np.linalg.norm(s - self.rbf_grid, axis=0) / 2)
        self.state_features = rbf
        return rbf

    def step(self, action):
        state, reward, done, info = super().step(action)
        info["state_features"] = self.get_features()
        ###########################
        ###  Added additional reward for being close to vertical
        ###########################
        reward += np.abs(state[2])/2
        return state, reward, done, info

    def get_sa_pairs(self):
        """
        Get state-action information (or "after-states") for all actions in the current state.

        :return: next_state_actions: List of length num_actions. Every list element
            is a 4-Tuple containing (
                state_features: a NumPy array containing the state-action features,
                reward: a float,
                done: a boolean indicating terminal state or not,
                info: a dict containing supplementary info (can be empty)
            )
        """
        current_state = self.state
        current_steps_beyond_done = self.steps_beyond_done
        current_state_features = self.state_features
        next_state_actions = []
        for a in self.discrete_to_list():
            next_state_actions.append(self.step(a))
            self.set_to_state(current_state, current_steps_beyond_done, current_state_features)
        return next_state_actions

    @staticmethod
    def get_choice_set_array(next_state_actions):
        """
        Takes in a list of action descriptions (i.e., the output of get_sa_pairs())
        and returns a choice_set: a 2D NumPy array containing the action features for
        all state-action pairs. Every row contains the action features of one action.

        (The choice_set array is a suitable input for agent.choose_action(choice_set))

        :param next_state_actions: List of length num_actions. Every list element
            is a 4-Tuple of (
                state_features: a list of floats containing the state-action features,
                reward: a float,
                done: a boolean indicating terminal state or not,
                info: a dict containing supplementary info (can be empty)
            )
        :return: choice_set, a Numpy array of dimensions (num_actions, num_features)
        """
        return np.vstack([action[3]["state_features"] for action in next_state_actions])

    def get_current_state(self):
        """return the info necessary to reset the environment to the present state"""
        return self.state, self.steps_beyond_done

    def set_to_state(self, state, steps_beyond_done, state_features=None):
        """reset the environment to a specified state"""
        self.state = state
        self.steps_beyond_done = steps_beyond_done
        if state_features is None:
            self.state_features = self.get_features()
        else:
            self.state_features = state_features

    def discrete_to_list(self, start=0):
        """Convert a discrete object as defined in AI gym to a list"""
        return list(range(start, self.num_actions))

    @staticmethod
    def generate_random_starting_states(num_states):
        """
        Generate a list of randomly sampled starting states
        :param num_states: int
        :return starting_states: list of NumPy arrays of length 4
        """
        starting_states = list(np.random.uniform(low=-0.05, high=0.05, size=(num_states, 4)))
        return starting_states
