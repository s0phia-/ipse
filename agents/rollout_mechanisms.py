import numpy as np


class BatchRollout:
    def __init__(self,
                 env,
                 rollout_length,
                 rollouts_per_action,
                 rollout_set_size,
                 stochastic_rollout_policy=True,
                 rollout_state_population=None,
                 # num_features,
                 # num_value_features,
                 gamma=0.99):
        self.name = "BatchRollout"
        self.env = env
        self.num_features = self.env.num_features
        self.num_actions = self.env.num_actions
        self.rollout_length = rollout_length
        self.rollouts_per_action = rollouts_per_action
        self.rollout_set_size = rollout_set_size
        self.stochastic_rollout_policy = stochastic_rollout_policy
        if rollout_state_population is None:
            self.rollout_state_population = self.env.generate_random_starting_states(self.rollout_set_size)
            self.rollout_set = self.rollout_state_population
        else:
            self.rollout_state_population = rollout_state_population
            indices = np.random.choice(a=len(self.rollout_state_population),
                                       size=self.rollout_set_size,
                                       replace=False if len(self.rollout_state_population) >= self.rollout_set_size else True)
            self.rollout_set = [self.rollout_state_population[i] for i in indices]

        self.gamma = gamma
        # if self.use_state_values:
        #     self.num_features = num_features
        #     self.num_value_features = num_value_features

    def construct_rollout_set(self):
        indices = np.random.choice(a=len(self.rollout_state_population),
                                   size=self.rollout_set_size,
                                   replace=False if len(self.rollout_state_population) >= self.rollout_set_size else True)
        self.rollout_set = [self.rollout_state_population[i] for i in indices]

    def perform_rollouts(self, policy):
        """
        Estimates action values using rollouts (where the rollout policy is given by the argument 'policy'.

        :param policy:
        :return rollouts: a dictionary containing
            - state_action_features, a 3D NumPy array of floats of shape (self.rollout_set_size, self.num_actions, self.num_features)
            - state_action_values, a 2D NumPy array of floats of shape (self.rollout_set_size, self.num_actions)
            - num_available_actions, a 1D NumPy array of floats of size self.rollout_set_size
            - did_rollout, a 1D NumPy array of floats of size self.rollout_set_size
        """
        self.construct_rollout_set()
        state_action_features = np.zeros((self.rollout_set_size, self.num_actions, self.num_features), dtype=np.float)
        state_action_values = np.zeros((self.rollout_set_size, self.num_actions), dtype=np.float)
        num_available_actions = np.zeros(self.rollout_set_size, dtype=np.int)
        did_rollout = np.ones(self.rollout_set_size, dtype=np.bool)
        for ix, rollout_state in enumerate(self.rollout_set):
            # if ix % 100 == 0:
            #     print(f"Rollout state = {ix}")

            # Set env to rollout starting state (assuming starting state is not 'done')
            self.env.set_to_state(state=rollout_state,
                                  steps_beyond_done=None)

            # Get and store state-action features / after-states
            next_state_actions = self.env.get_sa_pairs()
            state_action_features[ix] = self.env.get_choice_set_array(next_state_actions)
            num_rollout_actions = len(next_state_actions)
            num_available_actions[ix] = num_rollout_actions
            action_value_estimates = np.zeros(num_rollout_actions)

            for starting_action in range(num_rollout_actions):
                # Reset env to rollout starting state (in principle not needed for the first action rollout)
                self.env.set_to_state(state=rollout_state, steps_beyond_done=None)

                # Take the action which is to be (rollout-)evaluated in this iteration
                starting_state, starting_reward, done, info = self.env.step(starting_action)
                starting_done = None if not done else True

                # Perform rollout(s) using current policy
                for rollout_ix in range(self.rollouts_per_action):
                    self.env.set_to_state(state=starting_state, steps_beyond_done=starting_done)
                    cumulative_reward = starting_reward
                    rollout_count = 0
                    done = starting_done
                    while not done and rollout_count < self.rollout_length:
                        next_state_actions = self.env.get_sa_pairs()
                        choice_set = self.env.get_choice_set_array(next_state_actions)
                        action = policy.choose_action(choice_set, stochastic=self.stochastic_rollout_policy)
                        state, reward, done, info = self.env.step(action)
                        cumulative_reward += (self.gamma ** rollout_count) * reward
                        rollout_count += 1

                    action_value_estimates[starting_action] += cumulative_reward

            action_value_estimates /= self.rollouts_per_action

            state_action_values[ix, :] = action_value_estimates

        rollouts = dict(state_action_features=state_action_features,
                        state_action_values=state_action_values,
                        num_available_actions=num_available_actions,
                        did_rollout=did_rollout)
        return rollouts
