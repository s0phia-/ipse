from agents.rbpi import create_rbpi

from envs.cartpole_rbf import CartPoleRBF

from run_files.create_rollout_population import create_rollout_population
from run_files.learn_and_evaluate import learn_and_evaluate
from run_files.show_policy import show_policy
from run_files.utils_run import create_run_directories, process_parameters, plot_learning_curve

import numpy as np


np.random.seed(123)
id = "_Rbpi"
run_id_path, models_path, results_path, plots_path = create_run_directories(id)


parameters = dict(
    # Run parameters
    num_learning_iterations=30,
    num_eval_runs=10,
    num_random_seeds=5,

    # Env parameters
    informative_rewards=True,
    max_eval_steps=2000,

    # Agent parameters
    stochastic_rollout_policy=True,
    rollout_length=30,
    rollouts_per_action=3,
    rollout_set_size=1,
    max_number_of_choice_sets=500,
    regularization=False,
    regularization_method="ridge",
    regularization_strength=0.0,  # 0.00001,
    regularization_strength_method="fixed"
)
p = process_parameters(parameters, run_id_path)

env = CartPoleRBF(informative_rewards=True)

rollout_state_population = create_rollout_population(env)

agent_create_fn = create_rbpi
agent_create_fn_args = dict(
    env=env,
    parameter_dict=p,
    rollout_state_population=rollout_state_population
)

mean_returns = learn_and_evaluate(
    env,
    agent_create_fn,
    agent_create_fn_args,
    p.num_learning_iterations,
    p.num_eval_runs,
    p.max_eval_steps,
    p.num_random_seeds
)

plot_learning_curve(plots_path, mean_returns, x_axis=None, suffix="rbpi")
# show_policy(env, agent, num_steps=p.max_eval_steps)

