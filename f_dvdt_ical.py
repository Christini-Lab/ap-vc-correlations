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
from matplotlib.patches import Ellipse


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
    fig = plt.figure(figsize=(6.5, 6))
    fig.subplots_adjust(.1, .12, .95, .95)

    axs = []

    #PANEL 1
    grid = fig.add_gridspec(6, 6, hspace=1, wspace=1.4)
    sub = grid[0:3, 0:3]
    corr_subgrid = sub.subgridspec(3, 3, hspace=.3)
    corr_axs = []
    for row in range(0, 3):
        for col in range(0, 3):
            ax = fig.add_subplot(corr_subgrid[row, col])
            if col != 0:
                ax.set_yticklabels('')

            corr_axs.append(ax)
    plot_i_ap_corr(corr_axs, 'dVdt')
    corr_axs[0].set_xticks([-150, -50])
    corr_axs[-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    axs.append(corr_axs[0])

    #PANEL 2 – RMP vs dVdt
    ax = fig.add_subplot(grid[:3, 3:])
    plot_dvdt_rmp(ax)
    axs.append(ax)

    #PANEL 3 – all APs
    ax = fig.add_subplot(grid[3:, :3])
    plot_ap(ax)
    axs.append(ax)

    #PANEL 4 – 
    ax = fig.add_subplot(grid[3:, 3:])
    plot_dvdt_ical(ax)
    axs.append(ax)

    letters = ['A', 'B', 'C', 'D']
    #for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v, ax_vc_v1, ax_vc_v2]):
    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_title(letters[i], y=.98, x=-.1)
        else:
            ax.set_title(letters[i], y=.99, x=-.1)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-ical_dvdt.pdf', transparent=True)

    plt.show()


def plot_i_ap_corr(corr_axs, feature):
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
            if i == 0:
                corr_axs[i].text(-110, 37, f'{seg_names[i]}', fontsize=8)

            regplot(x=i_curr, y=valid_ap_dat[feature].values, ax=corr_axs[i], color='grey')
        corr_axs[i].spines['top'].set_visible(False)
        corr_axs[i].spines['right'].set_visible(False)
        ap_rng = valid_ap_dat[feature].max() - valid_ap_dat[feature].min()
        corr_axs[i].set_ylim(valid_ap_dat[feature].min()-ap_rng*.1, valid_ap_dat[feature].max()+ap_rng*.3)

    corr_axs[3].set_ylabel(feature_names[feature])


def plot_ap(ax):
    plot_cc(ax, 'spont', 'k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Voltage (mV)')


def plot_cc(ax, cc_type, col=None):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 650, 6500)

    n = 0

    for f in all_files:
        if '.DS' in f:
            continue

        ap_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')
        ap_dat, cc_shape = get_ap(ap_dat)

        if cc_shape == cc_type:
            if cc_type == 'spont':
                if col is None:
                    ax.plot(t, ap_dat, '#377eb8', alpha = .3, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = .3, rasterized=True)
                t_sp = t
                ap_dat_sp = ap_dat
            else:
                if col is None:
                    ax.plot(t, ap_dat, '#ff7f00', alpha = .5, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = .5, rasterized=True)

            if f == '4_021921_1_alex_control':
                ax.plot(t, ap_dat, 'red', rasterized=True)
            if f == '3_021121_1_alex_cisapride':
                ax.plot(t, ap_dat, 'red', rasterized=True)


            n += 1
        else:
            continue

    #if cc_type == 'spont':
    #    ax.plot(t_sp, ap_dat_sp, 'pink')

    #ax.text(100, 40, f'n={n}')
    ax.set_ylim(-75, 50)
    ax.set_xlabel('Time (ms)')


def get_ap(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    if (((v.max() - v.min()) < 20) or (v.max() < 0)):
        return v[60000:62500], 'flat'

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')

    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return v[0:2500], 'flat'

    idx_start = peak_idxs[0] - 150
    idx_end = idx_start + 6500

    return v[idx_start:idx_end], 'spont'


def plot_dvdt_rmp(ax):
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    valid_indices = np.invert(np.isnan(all_ap_features['dVdt']))

    ax.scatter(all_ap_features['MP'].values[valid_indices],
                    all_ap_features['dVdt'].values[valid_indices],
                    color='k')
    ax.set_xlabel('MDP (mV)')
    ax.set_ylabel(r'$dV/dt_{max}$')

    mask = all_ap_features['MP'] < -70

    ax.scatter(all_ap_features['MP'].values[mask],
                    all_ap_features['dVdt'].values[mask],
                    facecolors='none', edgecolors='r', marker='s', s=60)
    
    x = pearsonr(all_ap_features['MP'].values[valid_indices],
                    all_ap_features['dVdt'].values[valid_indices])

    el = Ellipse(xy=(-54, 8), width=28, height=18, angle=-25, edgecolor='grey', alpha=.5, linestyle='--', fc='none')

    ax.add_patch(el)

    regplot(all_ap_features['MP'].values[valid_indices],
                all_ap_features['dVdt'].values[valid_indices],
                ax=ax, color='k', ci=None)

    print(f'Correlation between dV/dt and MDP is: {x}')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 40)
    ax.text(-72, 38, r'MDP vs. $dV/dt_{max}$, ' + f'R={round(x[0], 2)}', fontsize=12)
    ax.set_xlim(-74.5, -42)


def plot_dvdt_ical(ax):
    feature = 'dVdt'
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

    correlation_time = 1986
    mid_idx = int(correlation_time*10)
    i_curr = valid_vc_dat[:, (mid_idx-10):(mid_idx+10)]
    i_curr = i_curr.min(1)

    ax.scatter(i_curr, valid_ap_dat['dVdt'].values, color='k')

    mask = valid_ap_dat[feature] > 20 

    ax.scatter(i_curr[mask], valid_ap_dat['dVdt'].values[mask], color='r', marker='x', s=60)
    regplot(x=i_curr[~mask], y=valid_ap_dat[feature].values[~mask], ax=ax, color='k', ci=None)

    feature_corrs = pearsonr(i_curr[~mask], valid_ap_dat[feature].values[~mask])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.set_title(r'$I_{CaL}$ segment, 'f'R={round(feature_corrs[0], 2)}', y=.9)
    ax.text(-15, 38, r'$I_{CaL}$ segment, 'f'R={round(feature_corrs[0], 2)}', fontsize=12)
    ax.set_ylim(0, 40)

    ax.set_xlabel(r'$I_{out}$ from $I_{CaL}$ segment (A/F)')
    ax.set_ylabel(r'$dV/dt_{max}$ (V/s)')


def main():
    plot_figure()

if __name__ == '__main__': 
    main()
