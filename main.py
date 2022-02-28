from dqn_style_run_files.evaluate import control_evaluation
from dqn_style_run_files.plot import plot_dict
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW

num_episodes = 500
max_length_episode = 100
sleep_every_step = 0
evaluate_every_x_episodes = 10
evaluate_iterations = 5
regularisation_strengths = [0, .01, .1, 1, 2, 5]


if __name__ == '__main__':
    all_returns = {}
    env = CartPoleRBF()
    for regularisation_cnst in regularisation_strengths:
        agent = QEW(num_features=env.num_features, actions=env.action_space, regularisation_strength=1, exploration=.15)
        returns = control_evaluation(agent, env, sleep_every_step,  num_episodes, max_length_episode,
                                     evaluate_every_x_episodes, evaluate_iterations)
        all_returns[regularisation_cnst] = returns
    env.close()
    try:
        del env
    except ImportError:
        pass
    plt.plot(all_returns)
    plt.show()
