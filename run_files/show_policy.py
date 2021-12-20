import time


def show_policy(env, agent, num_steps=200, time_sleep=0.02):
    env.reset()
    env.render()
    for _ in range(num_steps):
        time.sleep(time_sleep)
        next_state_actions = env.get_sa_pairs()
        choice_set = env.get_choice_set_array(next_state_actions)
        action = agent.choose_action(choice_set, stochastic=False)
        _, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()
