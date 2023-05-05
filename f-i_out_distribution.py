from os import listdir
import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from seaborn import histplot, distplot

import myokit

from utility_classes import VCSegment, VCProtocol

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_figure():
    fig = plt.figure(figsize=(6.5, 6))
    fig.subplots_adjust(.1, .07, .95, .95)

    #grid = fig.add_gridspec(3, 2, hspace=.32, wspace=0.1, height_ratios=[3, 4, 4])
    grid = fig.add_gridspec(2, 2, hspace=.32, wspace=0.1, height_ratios=[3, 4])
    correlation_times = [600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]


    ax_spont = fig.add_subplot(grid[0, 0])
    ax_flat = fig.add_subplot(grid[0, 1])


    plot_cc(ax_spont, 'spont')
    plot_cc(ax_flat, 'flat')

    vc_subgrid = grid[1,:].subgridspec(2, 1, wspace=.9, hspace=.1, height_ratios=[2,3])
    ax_vc_v = fig.add_subplot(vc_subgrid[0])
    ax_vc_i = fig.add_subplot(vc_subgrid[1])

    #zoom_subgrid = grid[2, :].subgridspec(2, 2, wspace=.2, hspace=.1)
    #ax_vc_v1 = fig.add_subplot(zoom_subgrid[0, 0])
    #ax_vc_v1.tick_params(labelbottom=False)
    #ax_vc_i1 = fig.add_subplot(zoom_subgrid[1, 0])

    #ax_vc_v2 = fig.add_subplot(zoom_subgrid[0, 1])
    #ax_vc_v2.tick_params(labelbottom=False)
    #ax_vc_i2 = fig.add_subplot(zoom_subgrid[1, 1])
    #windows= [[400, 900], [4000, 6000]]
    #zoom_axes = [[ax_vc_v1, ax_vc_i1], [ax_vc_v2, ax_vc_i2]]
    windows = []
    zoom_axes = []

    plot_vc(ax_vc_v, ax_vc_i, windows, zoom_axes)
    plot_mods(ax_vc_i, windows, zoom_axes)
    #plot_kernik_cc(ax_spont)

    #axs = [ax_spont, ax_flat, ax_vc_v, ax_vc_i, ax_vc_v1, ax_vc_i1, ax_vc_v2, ax_vc_i2]
    axs = [ax_spont, ax_flat, ax_vc_v, ax_vc_i]

    [ax_vc_v.axvline(curr_t, color='grey', linestyle='--', alpha=.5) for curr_t in correlation_times]
    seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']
    [ax_vc_v.text(curr_t-20, 70, seg_names[i]) for i, curr_t in enumerate(correlation_times)]


    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    letters = ['A', 'B', 'C', 'D', 'E']
    #for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v, ax_vc_v1, ax_vc_v2]):
    for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v]):
        if i == 2:
            ax.set_title(letters[i], y=.98, x=-.05)
        else:
            ax.set_title(letters[i], y=.98, x=-.1)


    ax_spont.set_ylabel('Voltage (mV)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-vc_dist_outlier.pdf', transparent=True)

    plt.show()


def plot_figure():
    fig, axs = plt.subplots(3, 3, figsize=(7.5, 6))
    fig.subplots_adjust(.1, .07, .95, .95, hspace=.3)

    axs = axs.flatten()
    
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    t_window = [4000, 13500]

    all_currs_dict = {}
    all_currs_list = []

    all_files = listdir('./data/cells')
    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10

    for i, f in enumerate(all_files):
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        cell_params = pd.read_excel(f'./data/cells/{f}/cell-params.xlsx')

        vc_curr = vc_dat['Current (pA/pF)'][start_idx:end_idx].values
        if f == '6_033021_4_alex_control':
            print(i)
            print(f)
            long_curr = vc_curr

        all_currs_dict[f] = vc_curr
        all_currs_list.append(vc_curr)

    times = vc_dat['Time (s)'][start_idx:end_idx]*1000 - t_window[0]
    times = times.values

    all_currs_df = pd.DataFrame(all_currs_dict)
    all_currs_df = all_currs_df.reindex(
            sorted(all_currs_df.columns), axis=1)

    correlation_times = [600, 1263, 1986, 2760, 3641, 4300, 5840, 9040]
    correlation_times = [1263, 1986, 2760, 3641, 4300, 5840, 9040]
    correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    seg_type = ['min', 'avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']

    seg_names = [r'$I_{Na1}$', r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']


    long_idx = 12

    for i, corr_time in enumerate(correlation_times):
        mid_idx = int(corr_time*10)
        i_curr = all_currs_df.values[(mid_idx-10):(mid_idx+10), :]
        long_curr_val = long_curr[(mid_idx-10):(mid_idx+10)]

        if seg_type[i] == 'avg':
            i_curr = i_curr.mean(0)
            long_curr_val = long_curr_val.mean()
        if seg_type[i] == 'min':
            i_curr = i_curr.min(0) 
            long_curr_val = long_curr_val.min()
        if seg_type[i] == 'max':
            i_curr = i_curr.max(0) 
            long_curr_val = long_curr_val.max()

        ax_num = i
        histplot(i_curr, ax=axs[i], color='k')
        axs[i].set_title(seg_names[i], y=.95)
        try:
            axs[i].axvline(long_curr_val, color='pink', linestyle='--', linewidth=2)
        except:
            import pdb
            pdb.set_trace()

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('')

    axs[3].set_ylabel('Count')
    axs[-2].set_xlabel(r'$I_{out}$ (A/F)')

    #axs[-1].axis('off')
    #axs[-3].axis('off')
    plt.savefig('./figure-pdfs/f-vc_dist_outlier.pdf', transparent=True)
    plt.show()






plot_figure()
