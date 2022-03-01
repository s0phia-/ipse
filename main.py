from dqn_style_run_files.evaluate import control_evaluation
from dqn_style_run_files.plot import plot_dict
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW

num_episodes = 200
max_length_episode = 200
sleep_every_step = 0
evaluate_every_x_episodes = 4
evaluate_iterations = 40
regularisation_strengths = [.0001, .1, 1]
regularisation_penalties = [True, False]


if __name__ == '__main__':
    all_returns = {}
    env = CartPoleRBF()
    for regularisation_const in regularisation_strengths:
        for regularisation_pen in regularisation_penalties:
            agent = QEW(num_features=env.num_features, actions=env.action_space,
                        regularisation_strength=regularisation_const, exploration=.15, ew=regularisation_pen)
            returns = control_evaluation(agent, env, sleep_every_step,  num_episodes, max_length_episode,
                                         evaluate_every_x_episodes, evaluate_iterations)
            if regularisation_pen is True:
                legend = 'EW, ' + str(regularisation_const)
            else:
                legend = 'Ridge, ' + str(regularisation_const)
            all_returns[legend] = returns
    env.close()
    try:
        del env
    except ImportError:
        pass
    plot_dict(all_returns, evaluate_every_x_episodes, "Regularisation")
