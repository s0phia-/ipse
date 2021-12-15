from envs.cartpole_rbf import CartPoleRBF
from envs.pub_or_study import StudentDilemma
import time

if __name__ == '__main__':
    sleep_every_step = 0.05  # 0

    # env = StudentDilemma()
    env = CartPoleRBF()
    env.reset()
    env.render()
    for _ in range(1000):
        time.sleep(sleep_every_step)
        action = env.action_space.sample()
        print(env.get_sa_pairs())
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()

