from datetime import datetime
import os
import numpy as np
import multiprocessing as mp
import argparse

from dqn_style_run_files.evaluate import full_run


if __name__ == '__main__':

    results_path = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pool = mp.Pool(mp.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--eval_every_x_episodes', type=int, default=5)
    parser.add_argument('--eval_iterations', type=int, default=3)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--reg_strengths', type=list, default=np.logspace(-3, 4, 20))
    parser.add_argument('--agents', type=list, default=["QStewAgentType1", "QRidgeAgentType1", "QEwAgentType1",
                                                        "RidgeAgent", "PureEwAgent", "StewAgent", "LinRegAgent"])
                                                        # "LspiAgent", "LspiAgentEw"])
    parser.add_argument('--direct_features', type=list, default=[False])
    args = parser.parse_args()

    all_run_args = [[agent_i, args.eval_every_x_episodes, args.eval_iterations, args.sleep, args.max_ep_len,
                     args.episodes, reg_coef, agent, df, results_path]
                    for agent_i in range(args.num_agents)
                    for reg_coef in args.reg_strengths
                    for agent in args.agents
                    for df in args.direct_features]
    # all_run_args = [[agent_i, 3, 3, 0, 200, 150, 4.6, "StewAgent", False, results_path] for agent_i in range(30)] + \
    #                [[agent_i, 3, 3, 0, 200, 150, 460, "RidgeAgent", False, results_path] for agent_i in range(30)] + \
    #                [[agent_i, 3, 3, 0, 200, 150, 1, "PureEwAgent", False, results_path] for agent_i in range(30)] + \
    #                [[agent_i, 3, 3, 0, 200, 150, 1, "LinRegAgent", False, results_path] for agent_i in range(30)]

    pool.starmap(full_run, all_run_args)
