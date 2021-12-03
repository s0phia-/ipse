from envs.cartpole_rbf import CartPoleRBF
from envs.pub_or_study import StudentDilemma

if __name__ == '__main__':
    env = CartPoleRBF()
    env.reset()
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()
        print(env.lookahead_one_step())
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()
