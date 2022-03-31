import matplotlib.pyplot as plt
import numpy as np
from utils import cross_axis_sd, cross_axis_mean
import plotnine as p9


def plot_dict(returns_dict, x_axis_factor, legend_label):
    for key, returns in returns_dict.items():
        y = cross_axis_mean(returns)
        x_axis = np.arange(0, len(y) * x_axis_factor, x_axis_factor)  # +(i*3)
        plt.plot(x_axis, y, label=key)
    plt.legend(title=legend_label)
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.show()


def plot_gg(pd_df):
    plot, fig = (p9.ggplot(data=pd_df,
                           mapping=p9.aes(x='episode', y='return', shape='agent', color='reg_coef')) +
                 p9.geom_line() +
                 p9.geom_point() +
                 p9.theme_bw()).draw(show=True, return_ggplot=True)
    return plot, fig
