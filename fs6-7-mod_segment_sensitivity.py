from os import listdir
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from sklearn import linear_model
from seaborn import heatmap, barplot
import seaborn as sns
from scipy.stats.stats import spearmanr

import matplotlib.pyplot as plt
import matplotlib

from seaborn import regplot

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)








i_segs = ['Ito', 'IK1', 'If', 'IKr']
i_segs = ['I6mV', 'IK1', 'If', 'IKr']
directory = 'pop_8_Kernik'

all_vc_dat = np.load(f'./data/mod_populations/{directory}/vc_iout.npy')
all_ind_params = pd.read_csv(f'./data/mod_populations/{directory}/all_params.csv')

all_ind_params = all_ind_params.drop(['voltageclamp.cm_est', 'voltageclamp.rseries_est'], axis=1)

correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
seg_type = ['min', 'avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']

current_indices = {'INa1':0, 'I6mV':1, 'IKr':2, 'ICaL':3, 'INa2':4, 'Ito':5, 'IK1':6, 'If':7, 'IKs':8}
seg_names = [r'$I_{Na1}$', r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

correlation_times = [correlation_times[current_indices[c]] for c in i_segs]
seg_type = [seg_type[current_indices[c]] for c in i_segs]

segment_corrs = {}

for i, corr_time in enumerate(correlation_times):
    params_corrs = []
    mid_idx = int(corr_time*10)
    i_curr = all_vc_dat[:, (mid_idx-10):(mid_idx+10)]

    if seg_type[i] == 'avg':
        i_curr = i_curr.mean(1)
    if seg_type[i] == 'min':
        i_curr = i_curr.min(1)
    if seg_type[i] == 'max':
        i_curr = i_curr.max(1)


    for param in all_ind_params.columns:
        params_corrs.append(spearmanr(all_ind_params[param].values, i_curr)[0])

    segment_corrs[i_segs[i]] = params_corrs

#fig = plt.figure(figsize=(3, 6))
fig = plt.figure(figsize=(4, 3))
fig.subplots_adjust(.13, .14, .95, .91)

#grid = fig.add_gridspec(2, 1, hspace=.3, wspace=.1)
grid = fig.add_gridspec(1, 1, hspace=.3, wspace=.1)

#fig, axs = plt.subplots(1, 4, figsize=(6.5, 5))
#fig.subplots_adjust(.1, .12, .95, .95)

#heat_ax = fig.add_subplot(grid[0])
subgrid = grid[0].subgridspec(1, 4, wspace=.4)

axs = [fig.add_subplot(subgrid[i]) for i in range(0, 4)]

feature = 'MP'
#heat_currents = None


names = [r'$g_{leak}$', 'Cm', r'$R_s$', r'$g_{Na}$', r'$g_{Kr}$', r'$g_{CaL}$',r'$g_{Ks}$',r'$g_{K1}$',r'$g_{f}$',r'$g_{to}$',r'$g_{NaK}$',r'$g_{NaCa}$',r'$g_{bNa}$',r'$g_{bCa}$',]
names = [r'$g_{leak}$', r'$g_{Na}$', r'$g_{Kr}$', r'$g_{CaL}$',r'$g_{Ks}$',r'$g_{K1}$',r'$g_{f}$',r'$g_{to}$',r'$g_{NaK}$',r'$g_{NaCa}$',r'$g_{bNa}$',r'$g_{bCa}$',]


for i, ax in enumerate(axs):
    x_vals = [segment_corrs[i_segs[i]][0]] + segment_corrs[i_segs[i]][3:]
    colors = ['r' if c < 0 else 'g' for c in x_vals]

    barplot(y=names, x=x_vals, palette=colors, orient='h', ax=ax, alpha=.7)
    ax.set_title(seg_names[current_indices[i_segs[i]]])
    ax.axvline(0, color='grey', alpha=.5, linestyle='--')
    [ax.axhline(i, color='grey', alpha=.2) for i in range(0, len(names))]
    ax.set_xlim(-1, 1)
    ax.set_xlabel('R')
    if i > 0:
        ax.tick_params(left=False)
        ax.set_yticks([])

    #ax.get_legend().remove()

for i in range(0, 4):
    rect = matplotlib.patches.Rectangle((-.9, -.4), 1.8, 1.8, facecolor='none', edgecolor='red', linestyle='--')
    #axs[i].add_patch(rect)

#letters = ['A', 'B']
##for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v, ax_vc_v1, ax_vc_v2]):
#for i, ax in enumerate([heat_ax, axs[0]]):
#    if i == 1:
#        ax.text(-2, -1, letters[i], fontsize=12)
#    else:
#        ax.set_title(letters[i], y=.98, x=-.1)


matplotlib.rcParams['pdf.fonttype'] = 42
plt.savefig('./figure-pdfs/f-curr_sensitivities.pdf', transparent=True)

plt.show()



