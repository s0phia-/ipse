from datetime import datetime
import os
import numpy as np
import multiprocessing as mp
import argparse

from dqn_style_run_files.evaluate import full_run

"""
 All agents:
QRidgeSeparatedAgent
QEwAgent
QStewSeparatedAgent
QLinRegSeparatedAgent
QStewTogetherAgent
QRidgeTogetherAgent
LspiAgent
LspiAgentL2
LspiAgentEw
QStewTogInc
QRidgeTogInc
QStewSepInc
QRidgeSepInc
QLinRegSepInc
"""

##############
# important! #
##############

# You can either run a full run with a cross product of all the different arguments parsed, or only run the agents for
# optimal regularisation parameters. Set to False for the former, True for the latter.
run_optimal = True

if __name__ == '__main__':

    results_path = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pool = mp.Pool(mp.cpu_count())

    #####################################
    # Run a cross product of parameters #
    #####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_optimal', type=bool, default=run_optimal)
    parser.add_argument('--num_agents', type=int, default=10)
    parser.add_argument('--eval_every_x_episodes', type=int, default=3)
    parser.add_argument('--eval_iterations', type=int, default=3)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--reg_strengths', type=list, default=np.logspace(-2, 1.5, 10))
    parser.add_argument('--agents', type=list, default=["QRidgeSeparatedAgent"])
    parser.add_argument('--direct_features', type=list, default=[True, False])
    args = parser.parse_args()

    run_optimal = args.run_optimal

    if not run_optimal:
        all_run_args = [[agent_i, args.eval_every_x_episodes, args.eval_iterations, args.sleep, args.max_ep_len,
                         args.episodes, reg_coef, agent, df, results_path]
                        for agent_i in range(args.num_agents)
                        for reg_coef in args.reg_strengths
                        for agent in args.agents
                        for df in args.direct_features]

    ############################################################
    # Option to run agents with optimal regression coefficient #
    ############################################################

    optimal_reg = {  # agent: [optimal regularisation strength, number of agents to compare]
        "QRidgeSeparatedAgent": [13, 30],
        "QEwAgent": [0, 30],
        "QStewSeparatedAgent": [4.6, 30],
        "QStewTogetherAgent": [2, 30],
        "QRidgeTogetherAgent": [0.06, 30],
        "QLinRegTogetherAgent": [0, 30],
        # "LspiAgent": [4.85, 20],
        # "LspiAgentL2": [100, 20],
        # "LspiAgentEw": [4.85, 20],
        "QStewTogInc": [0.035, 100],
        "QRidgeTogInc": [0.1, 100],
        "QStewSepInc": [0.007, 100],
        "QRidgeSepInc": [1, 100],
        "QLinRegSepInc": [0, 100]
    }

    if run_optimal:
        all_run_args = []
        for key, item in optimal_reg.items():
            all_run_args += [[agent_i, 3, 3, 0, 200, 500, item[0], key, True, results_path] for agent_i in
                             range(item[1])]

    pool.starmap(full_run, all_run_args)
