import numpy as np
import time

from gym.envs.classic_control import CartPoleEnv
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW_Together import QStewTogetherAgent


def evaluate(agent, env, sleep_time, episodes, max_episode_length):
    for _ in range(episodes):
        env.reset()
        env.render(mode="human")
        state = env.state_features
        for i in range(max_episode_length):
            action = agent.epsilon_greedy(state)
            env.render(mode="human")
            time.sleep(sleep_time)
            _, reward, done, info = env.step(action)
            state_prime = info["state_features"]
            state = state_prime
            if done or i == max_episode_length-1:
                break
    env.close()


if __name__ == '__main__':
    env = CartPoleRBF(direct_features=False)
    agent = QStewTogetherAgent(env.num_features, env.action_space, regularisation_strength=2)
    print('Untrained')
    evaluate(agent=agent, env=env, sleep_time=0.1, episodes=50,
             max_episode_length=200)
    agent.run(env=env, episodes=500, max_episode_length=200,
              sleep_time=0, stopping_criteria=1)
    print('Trained')
    evaluate(agent=agent, env=env, sleep_time=0.1, episodes=50,
             max_episode_length=200)
