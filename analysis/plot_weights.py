import numpy as np
import plotnine as p9
import pandas as pd


def plot_weights(pd_df, legend_pos="Bottom"):
    fig, plot = (p9.ggplot(data=pd_df,
                           mapping=p9.aes(x='index', y='weight', color='feature')) +
                 p9.geom_line() +
                 p9.theme_bw() +
                 p9.labs(x="Steps", y="Weights", color="Features")
                 ).draw(show=True, return_ggplot=True)
    return plot, fig

file_location = "../results/hello/QStewTogetherAgent_2_True_0.npy"
weights_memory = pd.DataFrame(np.load(file_location))
weights_memory['index'] = range(0, len(weights_memory))
long_weights_memory = pd.melt(weights_memory, var_name='feature', value_name='weight', id_vars=['index'])
print(long_weights_memory)
plot, fig = plot_weights(long_weights_memory)
fig



#
# file_location = "../results/look/"
# suffix = "QStewTogetherAgent_2.1544346900318843"
# all_weights = pd.DataFrame()
# for thing in os.listdir(file_location):
#     if thing[:37] != suffix:
#         continue
#     else:
#         weights_i = pd.DataFrame(np.load(file_location + thing))
#         weights_i['index'] = range(0, len(weights_i))
#         all_weights = pd.concat([all_weights, weights_i], axis=0)
# all_weights_mean = all_weights.groupby('index').mean().reset_index()
# long_weights_mean = pd.melt(all_weights_mean, var_name='feature', value_name='weight', id_vars=['index'])
# plot, fig = plot_weights(long_weights_mean)
# fig
