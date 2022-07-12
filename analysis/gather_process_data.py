import numpy as np
import os
import pandas as pd


def get_data(folder_path, eval_every_x_episodes):
    """
    Load every result file produced on a particular run into a Pandas DataFrame
    """
    runs = os.listdir(folder_path)
    all_returns = pd.DataFrame()
    for run in runs:
        returns = np.load(folder_path + '/' + run, allow_pickle=True)
        agent, reg_coef, directed_features, agent_number = run[:-4].split('_')
        n = len(returns)
        returns_one_run = {'agent': [agent] * n,
                           'reg_coef': [float(reg_coef)] * n,
                           'directed_features': [directed_features] * n,
                           'agent_number': [agent_number] * n,
                           'episode': range(eval_every_x_episodes, (n + 1) * eval_every_x_episodes,
                                            eval_every_x_episodes),
                           'return': returns}
        returns_one_run = pd.DataFrame(returns_one_run)
        all_returns = pd.concat([all_returns, returns_one_run], axis=0)
    #  all_returns = all_returns[all_returns['episode'] < 151]
    all_returns['reg_coef'] = all_returns['reg_coef'].astype(str)
    return all_returns


def process_data(df, compare_agents=None):
    if compare_agents is not None:
        df = df[df.agent.isin(compare_agents.keys())]
        df = df.replace({"agent": compare_agents})
    processed_data = df.groupby(['agent', 'reg_coef', 'episode'], as_index=False)['return'].agg(
        {'mean': 'mean', 'std': 'std'}).reset_index()
    processed_data['ymin'] = processed_data['mean'] - processed_data['std']
    processed_data['ymax'] = processed_data['mean'] + processed_data['std']
    print(processed_data)
    return processed_data


def find_best_reg_coef(df):
    df['reg_coef'] = df['reg_coef'].astype(float)
    avg_return = df.groupby(['agent', 'reg_coef'])['return'].mean()
    avg_return = pd.DataFrame({'return': avg_return}).reset_index()
    return avg_return
