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
        returns = np.load(folder_path + '/' + run)
        agent, reg_coef, directed_features, agent_number = run[:-4].split('_')
        n = len(returns)
        returns_one_run = {'agent': [agent]*n,
                           'reg_coef': [float(reg_coef)]*n,
                           'directed_features': [directed_features]*n,
                           'agent_number': [agent_number]*n,
                           'episode': range(eval_every_x_episodes, (n+1)*eval_every_x_episodes, eval_every_x_episodes),
                           'return': returns}
        returns_one_run = pd.DataFrame(returns_one_run)
        all_returns = pd.concat([all_returns, returns_one_run], axis=0)
        #all_returns['agent'] = np.where(all_returns['agent_rough'] == "ew",
        #                                "Equal Weights Regularised Linear Regression",
        #                                "Ridge Regression")
    return all_returns


def process_data(df):
    processed_data = df.groupby(['agent', 'reg_coef', 'episode'])['return'].mean()
    processed_data = pd.DataFrame({'return': processed_data}).reset_index()
    return processed_data


def find_best_reg_coef(df):
    avg_return = df.groupby(['agent', 'reg_coef'])['return'].mean()
    avg_return = pd.DataFrame({'return': avg_return}).reset_index()
    return avg_return
