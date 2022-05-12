from datetime import datetime
import os
import numpy as np
import multiprocessing as mp
import argparse

from dqn_style_run_files.evaluate import full_run

"""
All agents:
QRidgeSeparatedAgent
QEwSeparatedAgent
QStewSeparatedAgent
QLinRegSeparatedAgent
QStewTogetherAgent
QRidgeTogetherAgent
QLinRegTogetherAgent
LspiAgent
LspiAgentL2
LspiAgentEw
QStewTogInc
QRidgeTogInc
QLinRegTogInc
QStewSepInc
QRidgeSepInc
QLinRegSepInc
"""

if __name__ == '__main__':

    results_path = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pool = mp.Pool(mp.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=10)
    parser.add_argument('--eval_every_x_episodes', type=int, default=5)
    parser.add_argument('--eval_iterations', type=int, default=3)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--reg_strengths', type=list, default=[.007, .5]) #np.append([0], np.logspace(-5, 0, 15)))
    parser.add_argument('--agents', type=list, default=["QRidgeSepInc", "QStewSepInc"])
    parser.add_argument('--direct_features', type=list, default=[False])
    args = parser.parse_args()

    all_run_args = [[agent_i, args.eval_every_x_episodes, args.eval_iterations, args.sleep, args.max_ep_len,
                     args.episodes, reg_coef, agent, df, results_path]
                    for agent_i in range(args.num_agents)
                    for reg_coef in args.reg_strengths
                    for agent in args.agents
                    for df in args.direct_features]

    ############################################################
    # Option to run agents with optimal regression coefficient #
    ############################################################

    optimal_reg = {
        # "QRidgeSeparatedAgent": 0.35,
        # "QEwSeparatedAgent": 0,
        # "QStewSeparatedAgent": 4.6,
        # "QLinRegSeparatedAgent": 0,
        # "QStewTogetherAgent": 2,
        # "QRidgeTogetherAgent": 2,
        # "QLinRegTogetherAgent": 0,
        # "LspiAgent": 4.85,
        "LspiAgentL2": [25, 30],
        # "LspiAgentEw": 4.85,
        "QStewTogInc": [0.035, 100],
        "QRidgeTogInc": [0.1, 100],
        "QLinRegTogInc": [0, 100],
        "QStewSepInc": [0.007, 100],
        "QRidgeSepInc": [1, 100],
        "QLinRegSepInc": [0, 100]
    }

    all_run_args = []
    for key, item in optimal_reg.items():
        all_run_args += [[agent_i, 3, 3, 0, 200, 500, item[0], key, False, results_path] for agent_i in range(item[1])]

    pool.starmap(full_run, all_run_args)
