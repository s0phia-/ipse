import matplotlib.pyplot as plt
import numpy as np
from utils import cross_axis_sd, cross_axis_mean


def plot_dict(returns_dict, x_axis_factor, legend_label):
    #i=0
    for key, returns in returns_dict.items():
        #y_error = cross_axis_sd(returns)
        y = cross_axis_mean(returns)
        x_axis = np.arange(0, len(y)*x_axis_factor, x_axis_factor)#+(i*3)
        plt.plot(x_axis, y, label=key)
        #plt.errorbar(x_axis, y, yerr=y_error, label=key)
        #i+=1
    plt.legend(title=legend_label)
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.show()


def plot_dict_zoomed(returns_dict, x_axis_factor, legend_label, episodes=50):
    for key, returns in returns_dict.items():
        y_error = cross_axis_sd(returns)[:episodes/x_axis_factor]
        y = cross_axis_mean(returns)[:episodes/x_axis_factor]
        x_axis = np.arange(0, len(y)*x_axis_factor, x_axis_factor)
        plt.errorbar(x_axis, y, yerr=y_error, label=key)
    plt.legend(title=legend_label)
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.show()


