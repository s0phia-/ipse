from envs.cartpole_rbf import CartPoleRBF


if __name__ == '__main__':
    env = CartPoleRBF()
    env.reset()
    for _ in range(20):
        env.render()
        action = env.action_space.sample()
        env.step(action)
        print(action)
        print(env.features)
    env.close()
