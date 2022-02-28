import matplotlib.pyplot as plt
import statistics
from dqn_style_run_files.evaluate import evaluate, learn_and_evaluate
from envs.cartpole_rbf import CartPoleRBF
from agents.QEW import QEW

num_episodes = 500
max_length_episode = 100
sleep_every_step = 0
evaluate_every_x_episodes = 10
evaluate_iterations = 5


if __name__ == '__main__':
    all_returns = []
    env = CartPoleRBF()
    agent = QEW(num_features=env.num_features, actions=env.action_space)
    for i in range(num_episodes/evaluate_every_x_episodes):
        learn_and_evaluate(agent, env, sleep_every_step, evaluate_every_x_episodes, max_length_episode)
        if i % evaluate_every_x_episodes == 0:
            returns = evaluate(agent, env, sleep_every_step, evaluate_iterations, max_length_episode)
            all_returns.append(statistics.mean(returns))
    env.close()
    try:
        del env
    except ImportError:
        pass
    plt.plot(all_returns)
    plt.show()
