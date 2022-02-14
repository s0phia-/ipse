import numpy as np


def create_rollout_population(env, num_runs=100, max_steps=1000):
    """
    Using random policy.

    :param env:
    :param num_runs:
    :param max_steps:
    :return: rollout_starting_states, list of NumPy arrays of length 4
    """
    rollout_starting_states = []
    for run_ix in range(num_runs):
        state = env.reset()
        rollout_starting_states.append(state)
        step = 0
        while env.steps_beyond_done is None and step < max_steps:
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            if not done:
                rollout_starting_states.append(state)
            step += 1
    return rollout_starting_states
