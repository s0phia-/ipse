import numpy as np


def evaluate(env, agent, num_runs, max_steps=np.inf):
    returns = np.zeros(num_runs)
    for run_ix in range(num_runs):
        env.reset()
        cumulative_reward = 0.
        step = 0
        while env.steps_beyond_done is None and step < max_steps:
            next_state_actions = env.get_sa_pairs()
            choice_set = env.get_choice_set_array(next_state_actions)
            action = agent.choose_action(choice_set, stochastic=False)
            _, reward, _, _ = env.step(action)
            cumulative_reward += reward
            step += 1
        returns[run_ix] = cumulative_reward
    return returns
