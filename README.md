# Equal Weights Regularised Algorithms

This directory contains linear function approximator algorithms regularised with equal weights, along with unregularised and L2 regularised counterparts for comparison.

The code is run from the `main.py` file. There are two meta options: you can either run using optimal regularisation strength for each agent, or you can run a cross product of various hyperparameters (such as the number of agents to run, how often to evaluate them, regularisation strength and which linear function approximator to use). This is decided by the variable `run_optimal`. 

The code can be run from the command line.
To run the code with optimal regression strengths, use:
`python3 main.py --run_optimal True`

To run the code using the cross product of hyperparameters, use:
`python3 main.py --run_optimal False`
followed by the hyper parameters you wish to set. These are:
- `--num_agents`: number of agents to compare
- `--eval_every_x_episodes`: how often to evaluate the agent performance
- `--eval_iterations`: how many evaluation runs to perform per agent
- `--sleep`: length of time for agent to sleep between taking actions
- `--max_ep_len`: number of time steps after which to automatically end the episode. CartPole is usually 200
- `--episodes`: how many episodes to learn for 
- `--reg_strengths`: list of regularisation strengths to try 
- `--agents`: list of linear function approximator methods to train
- `--direct_features`: a list of booleans. True will multiply state feature values by -1 if the feature is correlated negatively with the outcome. False does nothing.  

The possible inputs to `--agents` are:
- `QRidgeSeparatedAgent`: Closed form ridge with separated state feature vectors for each action
- `QEwAgent`: Closed form pure equal weights: all weights set to 1
- `QStewSeparatedAgent`: Closed form equal weights regularised linear regression  with separated features for actions
- `QLinRegSeparatedAgent`: Closed form unregularised linear regression, separated state feature vectors for actions
- `QStewTogetherAgent`: Closed form equal weights regularised linear regression, 1 state action vector
- `QRidgeTogetherAgent`: Closed form ridge with 1 state action vector
- `QLinRegTogetherAgent`: Closed form unregularised linear regression, 1 state action vector 
- `LspiAgent`: Unregularised LSPI
- `LspiAgentL2`: L2 regularised LSPI
- `LspiAgentEw`: Equal weights regularised LSPI
- `QStewTogInc`: Incremental update equal weights regularised linear regression, 1 state action vector
- `QRidgeTogInc`: Incremental update ridge, 1 state action vector
- `QStewSepInc`: Incremental update equal weights regularised linear regression,separate state feature vector for actions
- `QRidgeSepInc`: Incremental update ridge, separate state feature vector for each action
- `QLinRegSepInc`: Incremental update unregularised linear regression

A full description of these agents is in the method section of the project report. Any agents not listed here are either not finished, or not my work (Malte Lichtenberg also created some agents in this repository, entirely separate from the agents mentioned here).
