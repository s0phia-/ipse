from analysis.gather_process_data import get_data, process_data, find_best_reg_coef
from analysis.plot import plot_gg, plot_coef_return

eval_every_x_episodes = 3
folder_path = '../results/hello'

all_returns = get_data(folder_path, eval_every_x_episodes)
all_returns = process_data(all_returns)

plot, fig = plot_gg(all_returns)
fig.savefig('image.png', dpi=300, bbox_inches='tight')

#####################################################
#  For finding the best regularisation coefficient  #
#####################################################
#
avg_returns = find_best_reg_coef(all_returns)
plot, fig = plot_coef_return(avg_returns)
fig.savefig('reg_image.png', dpi=300, bbox_inches='tight')

best = avg_returns.loc[avg_returns.groupby(by='agent')['return'].idxmax()]
print(best)
