from os import listdir
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from sklearn import linear_model
from seaborn import heatmap
from scipy.stats.stats import pearsonr


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


def plot_figure():
    fig = plt.figure(figsize=(6.5, 7))
    fig.subplots_adjust(.1, .07, .95, .95)

    #correlation_times = [600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    #correlation_times = [600, 1263, 1986, 2760, 3641, 4300, 5840, 9040]

    grid = fig.add_gridspec(6, 6, hspace=1, wspace=1.4)

    features = ['MP', 'APD90', 'dVdt']
    for i, sub in enumerate([grid[0:3, 0:3], grid[0:3, 3:], grid[3:, 0:3]]):
        feature = features[i]
        corr_subgrid = sub.subgridspec(3, 3, hspace=.3)
        corr_axs = []
        for row in range(0, 3):
            for col in range(0, 3):
                ax = fig.add_subplot(corr_subgrid[row, col])
                if col != 0:
                    ax.set_yticklabels('')

                corr_axs.append(ax)

        if i == 2:
            heatmap_ax = fig.add_subplot(grid[3:, 3:])
            plot_i_ap_corr(corr_axs, feature, heatmap_ax)
        else:
            plot_i_ap_corr(corr_axs, feature)

        #if i == 0:
        #    corr_axs[3].set_xlim(-200, 0)
        corr_axs[-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-exp_ap_vc_curr_corrs.pdf', transparent=True)

    plt.show()


def plot_i_ap_corr(corr_axs, feature, heatmap_ax=None):
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    t_window = [4000, 13500]

    all_currs_dict = {}
    all_currs_list = []

    all_files = listdir('./data/cells')
    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        cell_params = pd.read_excel(f'./data/cells/{f}/cell-params.xlsx')

        vc_curr = vc_dat['Current (pA/pF)'][start_idx:end_idx].values
        print(f)

        all_currs_dict[f] = vc_curr
        all_currs_list.append(vc_curr)

    times = vc_dat['Time (s)'][start_idx:end_idx]*1000 - t_window[0]
    times = times.values

    all_currs_df = pd.DataFrame(all_currs_dict)
    all_currs_df = all_currs_df.reindex(
            sorted(all_currs_df.columns), axis=1)
    all_ap_features = all_ap_features.sort_values('File')

    valid_indices = np.invert(np.isnan(all_ap_features[feature]))
    valid_ap_dat= all_ap_features[valid_indices]
    valid_vc_dat = np.array([all_currs_df[col_name] for col_name in valid_ap_dat['File']])

    all_curr_times = []

    #seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']
    #seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{Na}$', r'$I_{CaL}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']
    #correlation_times = [600, 1263, 1973, 1986, 3641, 4300, 5840, 9040]
    seg_names = [r'$I_{Na1}$', r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    seg_type = ['min', 'avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']



    feature_cols = {'MP': '#4daf4a', 'APD90': 'purple', 'dVdt': 'orange'}
    feature_names = {'MP': 'MP (mV)', 'APD90': r'$APD_{90}$ (ms)', 'dVdt': r'$dV/dt_{max}$ (V/s)'}

    for i, corr_time in enumerate(correlation_times):
        #if seg_names[i] == 'dIKs':
        #    i_curr1 = valid_vc_dat[:, int(corr_time*10)]
        #    i_curr2 = valid_vc_dat[:, int(9040*10)]

        #    i_curr = i_curr2 - i_curr1

        #else:
        mid_idx = int(corr_time*10)
        i_curr = valid_vc_dat[:, (mid_idx-10):(mid_idx+10)]
        if seg_type[i] == 'avg':
            i_curr = i_curr.mean(1)
        if seg_type[i] == 'min':
            i_curr = i_curr.min(1) 
        if seg_type[i] == 'max':
            i_curr = i_curr.max(1) 

        feature_corrs = pearsonr(valid_ap_dat[feature].values, i_curr)

        all_curr_times.append(i_curr)

        if feature_corrs[1] < 0.05:
            corr_axs[i].set_title(f'{seg_names[i]}, r={round(feature_corrs[0], 2)}', fontsize=8, y=.75)
            regplot(x=i_curr, y=valid_ap_dat[feature].values, ax=corr_axs[i], color=feature_cols[feature])
        else:
            corr_axs[i].set_title(f'{seg_names[i]}', fontsize=8, y=.77)
            regplot(x=i_curr, y=valid_ap_dat[feature].values, ax=corr_axs[i], color='grey')
        corr_axs[i].spines['top'].set_visible(False)
        corr_axs[i].spines['right'].set_visible(False)
        ap_rng = valid_ap_dat[feature].max() - valid_ap_dat[feature].min()
        corr_axs[i].set_ylim(valid_ap_dat[feature].min()-ap_rng*.1, valid_ap_dat[feature].max()+ap_rng*.3)

    corr = np.corrcoef(all_curr_times)
    if heatmap_ax is not None:
        heatmap(corr, ax=heatmap_ax, annot=True, annot_kws={"size":6}, mask=np.triu(corr), xticklabels=seg_names,yticklabels=seg_names, fmt='.1f')#, cbar=False)

    corr_axs[3].set_ylabel(feature_names[feature])


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]
    return np.array(new_vals)


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]
    return np.array(new_vals)


def main():
    #ap_vc_corr()
    #ap_vc_corr_singlet(5500)
    plot_figure()
    

if __name__ == '__main__':
    main()

