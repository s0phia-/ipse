import matplotlib.pyplot as plt
import numpy as np


def plot_dict(returns_dict, x_axis_factor, legend_label):
    for key, returns in returns_dict.items():
        x_axis = np.arange(0, len(returns)*x_axis_factor, x_axis_factor)
        plt.plot(x_axis, returns, label=key)
    plt.legend(title=legend_label)
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.show()