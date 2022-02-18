import time
import matplotlib.pyplot as plt

from envs.cartpole_rbf import CartPoleRBF
from envs.pub_or_study import StudentDilemma
from agents.QEW import QEW

num_episodes = 500
max_length_episode = 100


if __name__ == '__main__':
    sleep_every_step = 0.05  # 0

    env = CartPoleRBF()

    all_returns = []

    for _ in range(num_episodes):
        env.reset()
        env.render()
        state = env.state_features
        cumulative_reward = 0

        for _ in range(max_length_episode):
            time.sleep(sleep_every_step)
            agent = QEW(num_features=env.num_features, actions=env.action_space)
            action = agent.choose_action(state)  # env.action_space.sample()
            _, reward, done, info = env.step(action)
            cumulative_reward += reward
            state_prime = info["state_features"]
            agent.learn(state, action, reward, state_prime)
            env.render()
            state = state_prime
            if done:
                all_returns.append([cumulative_reward])
                break

    env.close()
    try:
        del env
    except ImportError:
        pass
    plt.plot(all_returns)
    plt.show()
