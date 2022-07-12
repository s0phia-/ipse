import time
import statistics
import numpy as np
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW_Separated import QRidgeSeparatedAgent, QStewSeparatedAgent, QLinRegSeparatedAgent, QStewSepInc, \
    QRidgeSepInc, QLinRegSepInc
from agents.QEW_Together import QEwAgent, QRidgeTogetherAgent, QStewTogetherAgent, QLinRegTogetherAgent, QStewTogInc, \
    QRidgeTogInc, QLinRegTogInc
from agents.LSPI import LspiAgent, LspiAgentEw, LspiAgentL2
# from agents.NN_QEW import DQNAgent


def evaluate(agent, env, sleep_time, episodes, max_episode_length):
    all_returns = []
    agent_epsilon = agent.epsilon
    agent.epsilon = 0
    for _ in range(episodes):
        env.reset()
        state = env.state_features
        cumulative_reward = 0
        for i in range(max_episode_length):
            time.sleep(sleep_time)
            action = agent.epsilon_greedy(state)
            _, reward, done, info = env.step(action)
            cumulative_reward += reward
            state_prime = info["state_features"]
            state = state_prime
            if done or i == max_episode_length-1:
                all_returns.append(cumulative_reward)
                break
    agent.epsilon = agent_epsilon
    return all_returns


def control_evaluation(agent, env, sleep_time, num_episodes, max_episode_length, evaluate_every_x_episodes,
                       evaluate_iterations, stopping_criteria):
    num_evaluations = int(round(num_episodes/evaluate_every_x_episodes))
    all_returns = []
    for i in range(num_evaluations):
        returns = evaluate(agent=agent, env=env, sleep_time=sleep_time, episodes=evaluate_iterations,
                           max_episode_length=max_episode_length)
        all_returns.append(statistics.mean(returns))
        agent.run(env=env, episodes=evaluate_every_x_episodes, max_episode_length=max_episode_length,
                  sleep_time=sleep_time, stopping_criteria=stopping_criteria)
    return all_returns, agent.weight_memory


def full_run(i_agent, evaluate_every_x_episodes, eval_iterations, sleep, max_ep_len, num_episodes, reg_coef, agent_name,
             direct_features, results_path, stopping_criteria=.005):
    env = CartPoleRBF(direct_features=direct_features)
    label = agent_name + '_' + str(reg_coef) + '_' + str(direct_features) + '_' + str(i_agent)
    print(label)
    agent = globals()[agent_name]
    agent = agent(env.num_features, env.action_space, regularisation_strength=reg_coef)
    returns, weights = control_evaluation(agent, env, sleep, num_episodes, max_ep_len, evaluate_every_x_episodes,
                                 eval_iterations, stopping_criteria)
    np.save(results_path + '_weights/' + label, weights)
    np.save(results_path + '/' + label, returns)
    return returns
