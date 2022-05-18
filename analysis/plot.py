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


def plot_gg(pd_df, legend_pos):
    fig, plot = (p9.ggplot(data=pd_df,
                           mapping=p9.aes(x='episode', y='mean', color='reg_coef', shape='agent')) +
                 p9.geom_line() +
                 p9.geom_point() +
                 p9.theme_bw() +
                 p9.labs(x="Episode", y="Total Return", color="Model") +
                 p9.geom_pointrange(p9.aes(ymin='ymin', ymax='ymax'), position=p9.position_dodge(width=1)) +
                 p9.theme(text=p9.element_text(size=20), legend_position=legend_pos)
                 ).draw(show=True, return_ggplot=True)
    return plot, fig


def plot_coef_return(df):
    fig, plot = (p9.ggplot(data=df,
                           mapping=p9.aes(x='reg_coef', y='return', color='agent')) +
                 p9.geom_point() +
                 p9.theme_bw()).draw(show=True, return_ggplot=True)
    return plot, fig
