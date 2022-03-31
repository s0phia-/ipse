import numpy as np
from dqn_style_run_files.evaluate import control_evaluation
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW
from agents.QEW_v2 import QEWv2


def full_run(i_agent, evaluate_every_x_episodes, eval_iterations, sleep, max_ep_len, num_episodes, reg_coef, agent,
             df, results_path):
    env = CartPoleRBF(direct_features=df)
    label = agent + '_' + str(reg_coef) + '_' + str(df) + '_' + str(i_agent)
    agent = QEWv2(num_features=env.num_features, actions=env.action_space,
                  regularisation_strength=reg_coef, model=agent)
    returns = control_evaluation(agent, env, sleep, num_episodes, max_ep_len, evaluate_every_x_episodes,
                                 eval_iterations)
    np.save(results_path + '/' + label, returns)
    return returns
