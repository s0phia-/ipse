import pickle

from dqn_style_run_files.evaluate import control_evaluation
from dqn_style_run_files.plot import plot_dict, plot_dict_zoomed
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW


num_episodes = 300
max_length_episode = 200
sleep_every_step = 0
evaluate_every_x_episodes = 10
evaluate_iterations = 5
regularisation_strengths = [1, .1]
function_approximators = ["ridge", "ew", "stew"]
agents_to_compare = 50

if __name__ == '__main__':
    all_returns = {}
    env = CartPoleRBF()
    for regularisation_const in regularisation_strengths:
        for ftn_appx in function_approximators:
            legend = ftn_appx + str(regularisation_const)
            agent_returns = []
            for _ in range(agents_to_compare):
                if 'agent' in globals():
                    del agent
                agent = QEW(num_features=env.num_features, actions=env.action_space,
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
    plot_dict_zoomed(all_returns, evaluate_every_x_episodes, "Regularisation")
