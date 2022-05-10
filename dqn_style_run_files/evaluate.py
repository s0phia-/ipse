import time
import statistics


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
    return all_returns
