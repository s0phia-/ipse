import numpy as np
from run_files.evaluate import evaluate


def learn_and_evaluate(
        env,
        agent_create_fn,
        agent_create_fn_args,
        num_learning_iterations,
        num_eval_runs,
        max_eval_steps,
        num_random_seeds,
        plot_results=False
):
    mean_returns = np.zeros((num_random_seeds, num_learning_iterations + 1))
    for random_seed in range(num_random_seeds):
        agent = agent_create_fn(**agent_create_fn_args)

        # First evaluation before any learning
        mean_returns[random_seed, 0] = np.mean(evaluate(env, agent, num_eval_runs, max_eval_steps))

        for learning_iteration in range(num_learning_iterations):
            agent.learn()
            returns = evaluate(env, agent, num_eval_runs, max_eval_steps)
            mean_return = np.mean(returns)
            print(f"Mean return = {mean_return}")
            mean_returns[random_seed, learning_iteration + 1] = mean_return

    print(f"mean_returns = {mean_returns}")
    return mean_returns
