import numpy as np
from dqn_style_run_files.evaluate import control_evaluation
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW_v2 import RidgeAgent, PureEwAgent, StewAgent, LinRegAgent
#from agents.NN_QEW import DQNAgent


def full_run(i_agent, evaluate_every_x_episodes, eval_iterations, sleep, max_ep_len, num_episodes, reg_coef, ftn_approx,
             direct_features, results_path):
    if ftn_approx == "ew":
        agent = PureEwAgent
    elif ftn_approx == "ridge":
        agent = RidgeAgent
    elif ftn_approx == "stew":
        agent = StewAgent
    else:
        agent = LinRegAgent
    env = CartPoleRBF(direct_features=direct_features)
    label = ftn_approx + '_' + str(reg_coef) + '_' + str(direct_features) + '_' + str(i_agent)
    agent = agent(env.num_features, env.action_space, regularisation_strength=reg_coef)
    returns = control_evaluation(agent, env, sleep, num_episodes, max_ep_len, evaluate_every_x_episodes,
                                 eval_iterations)
    np.save(results_path + '/' + label, returns)
    return returns
