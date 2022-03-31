from analysis.gather_process_data import get_data, process_data
from analysis.plot import plot_gg
import plotnine as p9


folder_path = '../results/runtime_2022-03-31 18:08:43.117989'

all_returns = get_data(folder_path)
all_returns = process_data(all_returns)
plot, fig = plot_gg(all_returns)

fig.savefig('image.png', dpi=300, bbox_inches='tight')

