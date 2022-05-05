import numpy as np
from dqn_style_run_files.evaluate import control_evaluation
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW_v2 import RidgeAgent, PureEwAgent, StewAgent, LinRegAgent
from agents.QEW import QStewAgentType1, QRidgeAgentType1, QEwAgentType1, QLinRegType1
from agents.LSPI import LspiAgent, LspiAgentEw
# from agents.NN_QEW import DQNAgent


def full_run(i_agent, evaluate_every_x_episodes, eval_iterations, sleep, max_ep_len, num_episodes, reg_coef, agent_name,
             direct_features, results_path, stopping_criteria=.5):
    env = CartPoleRBF(direct_features=direct_features)
    label = agent_name + '_' + str(reg_coef) + '_' + str(direct_features) + '_' + str(i_agent)
    print(label)
    agent = globals()[agent_name]
    agent = agent(env.num_features, env.action_space, regularisation_strength=reg_coef)
    returns = control_evaluation(agent, env, sleep, num_episodes, max_ep_len, evaluate_every_x_episodes,
                                 eval_iterations, stopping_criteria)
    np.save(results_path + '/' + label, returns)
    return returns
