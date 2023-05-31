from os import listdir
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from sklearn import linear_model
from seaborn import heatmap
from scipy.stats.stats import pearsonr 
from scipy import stats
import statsmodels.api as sm


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


def plot_figure_exp_vc_feature(feature, heat_currents):
    fig = plt.figure(figsize=(6.5, 4))
    fig.subplots_adjust(.1, .12, .95, .95)

    grid = fig.add_gridspec(3, 6, hspace=1, wspace=1.4)

    sub = grid[0:3, 0:3]
    corr_subgrid = sub.subgridspec(3, 3, hspace=.3)
    corr_axs = []

    for row in range(0, 3):
        for col in range(0, 3):
            ax = fig.add_subplot(corr_subgrid[row, col])
            if col != 0:
                ax.set_yticklabels('')

            corr_axs.append(ax)

    heatmap_ax = fig.add_subplot(grid[0:, 3:])
    plot_i_ap_corr(corr_axs, feature, heatmap_ax, heat_currents)
    corr_axs[-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig(f'./figure-pdfs/f-exp_ap_vc_curr_corrs_{feature}.pdf', transparent=True)

    plt.show()


def plot_mod_vc_feature(feature, heat_currents):
    fig = plt.figure(figsize=(6.5, 6))
    fig.subplots_adjust(.1, .07, .95, .95)

    grid = fig.add_gridspec(6, 6, hspace=1.2, wspace=1.4)

    all_corr_axs = []

    for i, sub in enumerate([grid[0:3, 3:], grid[3:, 3:]]):
        corr_subgrid = sub.subgridspec(3, 3, hspace=.5)
        corr_axs = []
        for row in range(0, 3):
            for col in range(0, 3):
                ax = fig.add_subplot(corr_subgrid[row, col])
                if col != 0:
                    ax.set_yticklabels('')

                corr_axs.append(ax)

        all_corr_axs.append(corr_axs)

    heatmap_ax = fig.add_subplot(grid[0:3, 0:3])
    heatmap_ax.set_title('Paci Simulations', y=.8)
    plot_i_ap_corr_mod(all_corr_axs[0], feature, 'pop_7_Paci', heatmap_ax, heat_currents)
    all_corr_axs[0][-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    heatmap_ax = fig.add_subplot(grid[3:, 0:3])
    heatmap_ax.set_title('Kernik Simulations', y=.8)
    plot_i_ap_corr_mod(all_corr_axs[1], feature, 'pop_8_Kernik', heatmap_ax, heat_currents)
    all_corr_axs[1][-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig(f'./figure-pdfs/f-mod_ap_vc_curr_corrs_{feature}.pdf', transparent=True)

    plt.show()


def plot_i_ap_corr(corr_axs, feature, heatmap_ax=None, heat_currents=None):
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    t_window = [4000, 13500]

    #all_ap_features = all_ap_features[all_ap_features.File != '4_021921_1_alex_control']
    #all_ap_features = all_ap_features[all_ap_features.File != '3_021121_1_alex_cisapride']

    all_currs_dict = {}
    all_currs_list = []

    all_files = listdir('./data/cells')
    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10

    for f in all_files:
        if '.DS' in f:
            continue
        #if '4_021921_1_alex_control' == f:
        #    continue
        #if '3_021121_1_alex_cisapride' == f:
        #    continue

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

    current_indices = {'INa1':0, 'I6mV':1, 'IKr':2, 'ICaL':3, 'INa2':4, 'Ito':5, 'IK1':6, 'If':7, 'IKs':8}

    seg_names = [r'$I_{Na1}$', r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    seg_type = ['min', 'avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']



    feature_cols = {'MP': '#4daf4a', 'APD90': 'purple', 'dVdt': 'orange'}
    feature_names = {'MP': 'MP (mV)', 'APD90': r'$APD_{90}$ (ms)', 'dVdt': r'$dV/dt_{max}$ (V/s)'}

    for i, corr_time in enumerate(correlation_times):
        mid_idx = int(corr_time*10)
        #if corr_time == 9040:
        #    i_curr = valid_vc_dat[:, (mid_idx-2000):(mid_idx+10)]
        #else:
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

    if heat_currents is not None:
        all_curr_times = [all_curr_times[current_indices[c]] for c in heat_currents]
        seg_names = [seg_names[current_indices[c]] for c in heat_currents]


    X = pd.DataFrame(np.transpose(np.array(all_curr_times)))
    Y = valid_ap_dat[feature].values

    X2 = sm.add_constant(X.values)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary())

    corr = np.corrcoef(all_curr_times)
    if heatmap_ax is not None:
        if heat_currents is not None:
            txt_fmt = '.2f'
        else:
            txt_fmt = '.1f'

        heatmap(corr, vmin=-.9, vmax=.9, ax=heatmap_ax, annot=True, annot_kws={"size":8}, mask=np.triu(corr), xticklabels=seg_names,yticklabels=seg_names, fmt=txt_fmt, cbar_kws = dict(use_gridspec=False,location="top", pad=.01), cmap="PiYG")

    corr_axs[3].set_ylabel(feature_names[feature])


def plot_i_ap_corr_mod(corr_axs, feature, directory, heatmap_ax=None, heat_currents=None):
    all_ap_features = pd.read_csv(f'./data/mod_populations/{directory}/all_ap_features.csv')
    all_vc_dat = np.load(f'./data/mod_populations/{directory}/vc_iout.npy')
    all_ap_features = all_ap_features.iloc[:75, :].copy()
    all_vc_dat = all_vc_dat[:75, :]


    seg_names = [r'$I_{Na1}$', r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    seg_type = ['min', 'avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']

    feature_cols = {'MP': '#4daf4a', 'APD90': 'purple', 'dVdt': 'orange'}
    feature_names = {'MP': 'MP (mV)', 'APD90': r'$APD_{90}$ (ms)', 'dVdt': r'$dV/dt_{max}$ (V/s)'}

    mask = ~all_ap_features[feature.lower()].isnull()

    all_curr_times = []

    for i, corr_time in enumerate(correlation_times):
        mid_idx = int(corr_time*10)
        i_curr = all_vc_dat[mask, (mid_idx-10):(mid_idx+10)]

        if seg_type[i] == 'avg':
            i_curr = i_curr.mean(1)
        if seg_type[i] == 'min':
            i_curr = i_curr.min(1) 
        if seg_type[i] == 'max':
            i_curr = i_curr.max(1) 

        feature_corrs = pearsonr(all_ap_features[feature.lower()].values[mask], i_curr)

        all_curr_times.append(i_curr)
        if feature_corrs[1] < 0.05:
            corr_axs[i].set_title(f'{seg_names[i]}, r={round(feature_corrs[0], 2)}', fontsize=8, y=.7)
            regplot(x=i_curr, y=all_ap_features[feature.lower()].values[mask], ax=corr_axs[i], color=feature_cols[feature])
        else:
            corr_axs[i].set_title(f'{seg_names[i]}', fontsize=8, y=.77)
            regplot(x=i_curr, y=all_ap_features[feature.lower()].values[mask], ax=corr_axs[i], color='grey')
        corr_axs[i].spines['top'].set_visible(False)
        corr_axs[i].spines['right'].set_visible(False)
        ap_rng = all_ap_features[feature.lower()].max() - all_ap_features[feature.lower()].min()
        corr_axs[i].set_ylim(all_ap_features[feature.lower()].min()-ap_rng*.1, all_ap_features[feature.lower()].max()+ap_rng*.3)

    current_indices = {'INa1':0, 'I6mV':1, 'IKr':2, 'ICaL':3, 'INa2':4, 'Ito':5, 'IK1':6, 'If':7, 'IKs':8}

    if heat_currents is not None:
        all_curr_times = [all_curr_times[current_indices[c]] for c in heat_currents]
        seg_names = [seg_names[current_indices[c]] for c in heat_currents]

    corr = np.corrcoef(all_curr_times)
    if heatmap_ax is not None:
        heatmap(corr, ax=heatmap_ax, annot=True, annot_kws={"size":8}, mask=np.triu(corr), xticklabels=seg_names,yticklabels=seg_names, fmt='.2f', cbar=False)

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
    #plot_figure()
    #plot_figure_exp_vc_feature('APD90', heat_currents=['I6mV', 'IK1', 'If', 'IKs'])
    #plot_mod_vc_feature('APD90', heat_currents=['I6mV', 'IK1', 'If', 'IKs'])

    plot_figure_exp_vc_feature('APD90', heat_currents=None)


if __name__ == '__main__':
    main()
