from agents.ipse import create_ipse
from agents.rbpi import create_rbpi

from envs.cartpole_rbf import CartPoleRBF

from run_files.create_rollout_population import create_rollout_population
from run_files.learn_and_evaluate import learn_and_evaluate
from run_files.utils_run import create_run_directories, process_parameters, plot_multiple_learning_curves

import numpy as np
import time
import os


np.random.seed(123)
exp_id = "_Comp"
run_id_path, models_path, results_path, plots_path = create_run_directories(exp_id)


parameters = dict(
    # Run parameters
    num_learning_iterations=50,
    num_eval_runs=20,
    num_random_seeds=10,

    # Env parameters
    informative_rewards=True,
    max_eval_steps=500,

    # Agent parameters
    stochastic_rollout_policy=True,
    rollout_length=30,
    rollouts_per_action=5,
    rollout_set_size=1,
    max_number_of_choice_sets=500,
    regularization=False,
    regularization_method="ridge",
    regularization_strength=0.0,  # 0.00001,
    regularization_strength_method="fixed",
    forced_phase_switch_after=20
)
p = process_parameters(parameters, run_id_path)

env = CartPoleRBF(informative_rewards=True)

rollout_state_population = create_rollout_population(env)


# Individual runs
compare_ids = ["rbpi", "ipse"]
compare_results = list()
compare_id_ix = 0


# RBPI
agent_create_fn = create_rbpi
agent_create_fn_args = dict(
    env=env,
    parameter_dict=p,
    rollout_state_population=rollout_state_population
)
time_total_begin = time.time()
mean_returns = learn_and_evaluate(
    env,
    agent_create_fn,
    agent_create_fn_args,
    p.num_learning_iterations,
    p.num_eval_runs,
    p.max_eval_steps,
    p.num_random_seeds
)
print(f"Total time passed: {time.time()-time_total_begin} seconds.")
np.save(file=os.path.join(results_path, f"test_results_{compare_ids[compare_id_ix]}.npy"),
        arr=mean_returns)
compare_results.append(mean_returns)
compare_id_ix += 1


# IPSE
agent_create_fn = create_ipse
time_total_begin = time.time()
mean_returns = learn_and_evaluate(
    env,
    agent_create_fn,
    agent_create_fn_args,
    p.num_learning_iterations,
    p.num_eval_runs,
    p.max_eval_steps,
    p.num_random_seeds
)
print(f"Total time passed: {time.time()-time_total_begin} seconds.")
np.save(file=os.path.join(results_path, f"test_results_{compare_ids[compare_id_ix]}.npy"),
        arr=mean_returns)
compare_results.append(mean_returns)
compare_id_ix += 1


plot_multiple_learning_curves(plots_path, compare_results, compare_ids, x_axis=None)
print(f"Results can be found in directory: {run_id_path}")


# show_policy(env, agent, num_steps=p.max_eval_steps)
