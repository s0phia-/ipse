import time


def evaluate(agent, env, sleep_time, episodes, max_episode_length):
    all_returns = []
    agent_epsilon = agent.epsilon
    agent.epsilon = 0
    for _ in range(episodes):
        env.reset()
        state = env.state_features
        cumulative_reward = 0

        for _ in range(max_episode_length):
            time.sleep(sleep_time)
            action = agent.choose_action(state)
            _, reward, done, info = env.step(action)
            cumulative_reward += reward
            state_prime = info["state_features"]
            state = state_prime
            if done:
                all_returns.append([cumulative_reward])
                break
    agent.epsilon = agent_epsilon
    return all_returns


def learn_and_evaluate(agent, env, sleep_time, episodes, max_episode_length):
    all_returns = []
    for i in range(episodes):
        env.reset()
        state = env.state_features
        cumulative_reward = 0
        for _ in range(max_episode_length):
            time.sleep(sleep_time)
            action = agent.choose_action(state)  # env.action_space.sample()
            _, reward, done, info = env.step(action)
            cumulative_reward += reward
            state_prime = info["state_features"]
            agent.learn(state, action, reward, state_prime)
            state = state_prime
            if done:
                all_returns.append([cumulative_reward])
                break
    return all_returns
