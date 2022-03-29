import pickle
import numpy as np

from dqn_style_run_files.evaluate import control_evaluation
from dqn_style_run_files.plot import plot_dict
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW
from agents.QEW_v2 import QEWv2


num_episodes = 150
max_length_episode = 200
sleep_every_step = 0
evaluate_every_x_episodes = 5
evaluate_iterations = 3
regularisation_strengths = np.logspace(-2, 1.5, 50)
function_approximators = ["stew", "ridge"]  # , "ew", "stew"]
agents_to_compare = 5

if __name__ == '__main__':
    all_returns = {}
    env = CartPoleRBF()
    for regularisation_const in regularisation_strengths:
        for ftn_appx in function_approximators:
            if ftn_appx in ['lin_reg', 'ew']:
                if regularisation_const == .1:
                    continue
                else:
                    legend = ftn_appx
            else:
                legend = ftn_appx + str(regularisation_const)
            agent_returns = []
            for _ in range(agents_to_compare):
                if 'agent' in globals():
                    del agent
                agent = QEWv2(num_features=env.num_features, actions=env.action_space,
                              regularisation_strength=regularisation_const, exploration=.15, model=ftn_appx)
                returns = control_evaluation(agent, env, sleep_every_step,  num_episodes, max_length_episode,
                                             evaluate_every_x_episodes, evaluate_iterations)
                agent_returns.append(returns)
            all_returns[legend] = agent_returns
    env.close()
    try:
        del env
    except ImportError:
        pass
    pickle.dump(all_returns, open("save_all_returns.p", "wb"))
    plot_dict(all_returns, evaluate_every_x_episodes, "Regularisation")
