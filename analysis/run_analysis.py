from analysis.gather_process_data import get_data, process_data, find_best_reg_coef
from analysis.plot import plot_gg, plot_coef_return

"""
All agents:
"QRidgeSeparatedAgent": "Ridge, separated actions",
"QEwAgent": "Pure equal weights",
"QStewSeparatedAgent": "Equal weights regularised, separated actions",
"QLinRegSeparatedAgent": "Unregularised, separated actions",
"QStewTogetherAgent": "Equal weights regularised, grouped actions",
"QRidgeTogetherAgent": "Ridge, grouped actions",
"QLinRegTogetherAgent": "Unregularised, grouped actions",
"LspiAgent": "LSPI",
"LspiAgentL2": "L2 regularised LSPI",
"LspiAgentEw": "Equal-weights regularised LSPI",
"QStewTogInc": "Equal weights regularised, grouped actions",
"QRidgeTogInc": "Ridge, grouped actions",
"QStewSepInc": "Equal weights regularised, separated actions",
"QRidgeSepInc": "Ridge, separated actions",
"QLinRegSepInc": "Unregularised"
"""
eval_every_x_episodes = 3
folder_path = '../results/all'

compare_agents = {"QStewSeparatedAgent": "Fit closed form, separated actions",
                  "QStewTogetherAgent": "Fit closed form, grouped actions",
                  "QStewSepInc": "Fit incrementally, separated actions",
                  "QStewTogInc": "Fit incrementally, grouped actions",
                  "LspiAgentEw": "LSPI"
                  }


all_returns = get_data(folder_path, eval_every_x_episodes)
all_returns = process_data(all_returns, compare_agents)

plot, fig = plot_gg(all_returns, "bottom")
fig.savefig('image1.png', dpi=300, bbox_inches='tight')

plot, fig = plot_gg(all_returns, "none")
fig.savefig('image2.png', dpi=300, bbox_inches='tight')

#####################################################
#  For finding the best regularisation coefficient  #
#####################################################

# avg_returns = find_best_reg_coef(all_returns)
# plot, fig = plot_coef_return(avg_returns)
# fig.savefig('reg_image.png', dpi=300, bbox_inches='tight')
#
# best = avg_returns.loc[avg_returns.groupby(by='agent')['return'].idxmax()]
# print(best)
#
hello = all_returns[all_returns['episode'] == max(all_returns['episode'])].groupby(by='agent')['return'].idxmax()
print(all_returns.loc[hello])
