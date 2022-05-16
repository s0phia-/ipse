import numpy as np
import time

from envs.cartpole_rbf import CartPoleRBF
from agents.QEW_Together import QStewTogetherAgent


def evaluate(agent, env, sleep_time, episodes, max_episode_length):
    all_returns = []
    agent_epsilon = agent.epsilon
    agent.epsilon = 0
    for _ in range(episodes):
        env.reset()
        env.render(mode="human")
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
    env.close()
    return all_returns


if __name__ == '__main__':
    env = CartPoleRBF(direct_features=False)
    agent = QStewTogetherAgent(env.num_features, env.action_space, regularisation_strength=2)
    print('Untrained')
    evaluate(agent=agent, env=env, sleep_time=0.01, episodes=50,
             max_episode_length=200)
    agent.run(env=env, episodes=500, max_episode_length=200,
              sleep_time=0, stopping_criteria=1)
    print('Trained')
    evaluate(agent=agent, env=env, sleep_time=0.01, episodes=50,
             max_episode_length=200)
