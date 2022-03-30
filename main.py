from datetime import datetime
import os
import numpy as np
import multiprocessing as mp
import argparse

from dqn_style_run_files.full_run import full_run


if __name__ == '__main__':

    results_path = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pool = mp.Pool(mp.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=30)
    parser.add_argument('--eval_every_x_episodes', type=int, default=3)
    parser.add_argument('--eval_iterations', type=int, default=3)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--reg_strengths', type=list, default=np.logspace(-2, 1.5, 50))
    parser.add_argument('--agents', type=list, default=["stew", "ridge"])  # , "ew", "lin_reg"])
    parser.add_argument('--direct_features', type=list, default=[False])
    args = parser.parse_args()

    all_run_args = [[agent_i, args.eval_every_x_episodes, args.eval_iterations, args.sleep, args.max_ep_len,
                     args.episodes, reg_coef, agent, df]
                    for agent_i in range(args.num_agents)
                    for reg_coef in args.reg_strengths
                    for agent in args.agents
                    for df in args.direct_features]
    pool.starmap(full_run, all_run_args)



    # for df in args.direct_features:
    #     env = CartPoleRBF(direct_features=df)
    #     for reg_coef in args.reg_strengths:
    #         for agent in args.agents:
    #             for i_agent in args.num_agents:
    #                 label = agent + str(reg_coef) + str(df) + str(i_agent)
    #                 agent = QEWv2(num_features=env.num_features, actions=env.action_space,
    #                               regularisation_strength=reg_coef, model=agent)
    #                 p = mp.Process(target=control_evaluation, args=(agent, env, args.sleep, args.episodes,
    #                                                                 args.max_ep_len, args.evaluate_every_x_episodes,
    #                                                                 args.eval_iterations, qq))
    #                 # returns = control_evaluation(agent, env, args.sleep, args.episodes, args.max_ep_len,
    #                 #                              args.evaluate_every_x_episodes, args.eval_iterations, qq)
    #                 np.save(results_path + '/' + label, returns)
    #     env.close()
    #


    # num_episodes = 150
    # max_length_episode = 200
    # sleep_every_step = 0
    # evaluate_every_x_episodes = 5
    # evaluate_iterations = 3
    # regularisation_strengths = [.1, 1, 10]  # np.logspace(-2, 1.5, 50)
    # function_approximators = ["stew"] #, "ridge", "ew", "lin_reg"]
    # agents_to_compare = 3

    # all_returns = {}
    # for tf in [True, False]:
    #     env = CartPoleRBF(direct_features=tf)
    #     for regularisation_const in regularisation_strengths:
    #         for ftn_appx in function_approximators:
    #             if ftn_appx in ['lin_reg', 'ew']:
    #                 if regularisation_const == .1:
    #                     continue
    #                 else:
    #                     legend = ftn_appx
    #             else:
    #                 legend = ftn_appx + str(regularisation_const) + str(tf)
    #             agent_returns = []
    #             for _ in range(agents_to_compare):
    #                 if 'agent' in globals():
    #                     del agent
    #                 agent = QEWv2(num_features=env.num_features, actions=env.action_space,
    #                               regularisation_strength=regularisation_const, exploration=.15, model=ftn_appx)
    #                 returns = control_evaluation(agent, env, sleep_every_step,  num_episodes, max_length_episode,
    #                                              evaluate_every_x_episodes, evaluate_iterations)
    #                 agent_returns.append(returns)
    #             all_returns[legend] = agent_returns
    #     env.close()
    #     try:
    #         del env
    #     except ImportError:
    #         pass
    #
    # pickle.dump(all_returns, open("save_all_returns.p", "wb"))
    # plot_dict(all_returns, evaluate_every_x_episodes, "Regularisation")
