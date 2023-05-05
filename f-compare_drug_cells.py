from os import listdir
import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import myokit

from utility_classes import VCSegment, VCProtocol

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


# CISAPRIDE IS NOT THE BEST
#4_022621_2_alex_cisapride - 95.2 -> 67.4
#4_022521_2_alex_cisapride - 92.1 -> 126.2 # NOT THE BEST

#5_031821_2_alex_verapamil - 137.5 -> 118.4
#5_031921_3_alex_verapamil - 213.7 -> 361.5 # NOT THE BEST

#7_042021_2_alex_quinine	57	    63.1
#7_042721_2_alex_quinine	129.6	101.2
#7_042721_4_alex_quinine	97.6	109.6
#7_042621_6_alex_quinine	170.8	267.7


def plot_figure():
    fig = plt.figure(figsize=(6.5, 6.5))
    fig.subplots_adjust(.1, .07, .95, .95)

    grid = fig.add_gridspec(4, 2, hspace=.5, wspace=0.25, width_ratios=[1, 2], height_ratios=[1,2,1,2])

    ax_spont1 = fig.add_subplot(grid[:2, 0])
    ax_spont2 = fig.add_subplot(grid[2:, 0])

    f1 = '7_042721_2_alex_quinine'
    f2 = '7_042721_4_alex_quinine'

    plot_cc_drug(ax_spont1, f1)
    plot_cc_drug(ax_spont2, f2)

    t_window = [4000, 13500]
    ax_vc_v1 = fig.add_subplot(grid[0, 1])
    ax_vc_v2 = fig.add_subplot(grid[2, 1])
    plot_v(ax_vc_v1, t_window)
    plot_v(ax_vc_v2, t_window)

    ax_vc_i1 = fig.add_subplot(grid[1, 1])
    ax_vc_i2 = fig.add_subplot(grid[3, 1])
    plot_i(ax_vc_i1, f1, t_window)
    ax_vc_i1.set_ylim(-15, 10)
    plot_i(ax_vc_i2, f2, t_window)
    ax_vc_i2.set_ylim(-15, 10)

    ax_vc_i1.set_ylabel('Current (A/F)')
    ax_vc_i2.set_ylabel('Current (A/F)')
    ax_vc_i2.set_xlabel('Time (ms)')

    axs = [ax_spont1, ax_spont2, ax_vc_v1, ax_vc_v2, ax_vc_i1, ax_vc_i2]
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    letters = ['A', 'B', 'C', 'D', 'E']
    for i, ax in enumerate([ax_spont1, ax_spont2]):
        if i == 2:
            ax.set_title(letters[i], y=.98, x=-.05)
        else:
            ax.set_title(letters[i], y=.98, x=-.1)

    ax_spont1.set_ylabel('Voltage (mV)')
    ax_spont2.set_ylabel('Voltage (mV)')
    ax_spont2.set_xlabel('Time (ms)')

    ax_spont1.legend()

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-ap_vc_drug_comparison.pdf', transparent=True)

    plt.show()


def plot_figure_big():
    fig = plt.figure(figsize=(6.5, 8))
    fig.subplots_adjust(.1, .07, .95, .95)

    grid = fig.add_gridspec(4, 2, hspace=.5, wspace=0.15, height_ratios=[3,2,4,4])

    f1 = '7_042721_2_alex_quinine'
    f2 = '7_042721_4_alex_quinine'

    axs = []

    #SPONTANEOUS
    ap_subgrid = grid[0,0].subgridspec(1, 2, width_ratios=[3, 1])
    ax_spont1 = fig.add_subplot(ap_subgrid[0])
    ax_up1 = fig.add_subplot(ap_subgrid[1])

    ap_subgrid = grid[0,1].subgridspec(1, 2, width_ratios=[3, 1])
    ax_spont2 = fig.add_subplot(ap_subgrid[0])
    ax_up2 = fig.add_subplot(ap_subgrid[1])

    ax_spont1.text(300, 30, 'Cell 1', fontsize=10)
    ax_spont2.text(300, 30, 'Cell 2', fontsize=10)

    t_length = 9000
    plot_cc_drug(ax_spont1, f1, t_length)
    plot_cc_drug(ax_spont2, f2, t_length)
    t_length = 200 
    plot_cc_drug(ax_up1, f1, t_length)
    plot_cc_drug(ax_up2, f2, t_length)

    axs = axs + [ax_spont1, ax_up1, ax_spont2, ax_up2]
    [ax.set_yticklabels([]) for ax in [ax_up1, ax_spont2, ax_up2]]
    ax_spont1.set_ylabel('Voltage (mV)')
    [ax.set_xlabel('Time (ms)') for ax in [ax_spont1, ax_spont2]]

    # VC
    t_window = [4000, 13500]
    ax_vc_v = fig.add_subplot(grid[1, :])
    plot_v(ax_vc_v, t_window)
    ax_vc_v.set_ylabel('Voltage (mV)')
    ax_vc_v.set_xlabel('Time (ms)')

    axs += [ax_vc_v]

    t_windows = []

    # VC BOX 1
    t_window = [400, 1000]
    t_windows += [t_window]
    vc_subgrid = grid[2,0].subgridspec(3, 1)
    ax_vc_v1  = fig.add_subplot(vc_subgrid[0])
    ax_vc_i11 = fig.add_subplot(vc_subgrid[1])
    ax_vc_i12 = fig.add_subplot(vc_subgrid[2])
    plot_v(ax_vc_v1, t_window, is_time_sub=True)
    plot_i(ax_vc_i11, f1, t_window, is_time_sub=True, ls='-')
    plot_i(ax_vc_i12, f2, t_window, is_time_sub=True, ls='-')
    ax_vc_i11.set_ylim(-5, 5)
    ax_vc_i12.set_ylim(-5, 8)
    axs += [ax_vc_v1, ax_vc_i11, ax_vc_i12]

    # VC BOX 2
    t_window = [1970, 1978]
    t_windows += [t_window]
    vc_subgrid = grid[2,1].subgridspec(3, 1)
    ax_vc_v2  = fig.add_subplot(vc_subgrid[0])
    ax_vc_i21 = fig.add_subplot(vc_subgrid[1])
    ax_vc_i22 = fig.add_subplot(vc_subgrid[2])
    plot_v(ax_vc_v2, t_window, is_time_sub=True)
    plot_i(ax_vc_i21, f1, t_window, is_time_sub=True, ls='-')
    #ax_vc_i21.set_ylim(-16, -5)
    #ax_vc_i22.set_ylim(-12, -2)
    plot_i(ax_vc_i22, f2, t_window, is_time_sub=True, ls='-')
    axs += [ax_vc_v2, ax_vc_i21, ax_vc_i22]

    # VC BOX 3
    #t_window = [3350, 3640]
    t_window = [3400, 3600]
    t_windows += [t_window]
    vc_subgrid = grid[3,0].subgridspec(3, 1)
    ax_vc_v3  = fig.add_subplot(vc_subgrid[0])
    ax_vc_i31 = fig.add_subplot(vc_subgrid[1])
    ax_vc_i32 = fig.add_subplot(vc_subgrid[2])
    plot_v(ax_vc_v3, t_window, is_time_sub=True)
    plot_i(ax_vc_i31, f1, t_window, is_time_sub=True, ls='-')
    ax_vc_i31.set_ylim(-8, 0)
    ax_vc_i32.set_ylim(-3, 3)
    plot_i(ax_vc_i32, f2, t_window, is_time_sub=True, ls='-')
    axs += [ax_vc_v3, ax_vc_i31, ax_vc_i32]

    # VC BOX 4
    t_window = [4800, 6000]
    t_windows += [t_window]
    vc_subgrid = grid[3,1].subgridspec(3, 1)
    ax_vc_v4  = fig.add_subplot(vc_subgrid[0]) 
    ax_vc_i41 = fig.add_subplot(vc_subgrid[1]) 
    ax_vc_i42 = fig.add_subplot(vc_subgrid[2]) 
    plot_v(ax_vc_v4, t_window, is_time_sub=True)
    plot_i(ax_vc_i41, f1, t_window, is_time_sub=True, ls='-')
    ax_vc_i41.set_ylim(-16, -5)
    ax_vc_i42.set_ylim(-12, -2)
    plot_i(ax_vc_i42, f2, t_window, is_time_sub=True, ls='-')
    axs += [ax_vc_v4, ax_vc_i41, ax_vc_i42]

    [ax.set_xticklabels([]) for ax in [ax_vc_v1, ax_vc_i11,
                                       ax_vc_v2, ax_vc_i21,
                                       ax_vc_v3, ax_vc_i31,
                                       ax_vc_v4, ax_vc_i41
                                       ]]

    [ax.set_ylabel('mV') for ax in [ax_vc_v1, ax_vc_v3]]
    [ax.set_ylabel('C1 A/F') for ax in [ax_vc_i11, ax_vc_i31]]
    [ax.set_ylabel('C2 A/F') for ax in [ax_vc_i12, ax_vc_i32]]

    [ax.set_xlabel('Time (ms)') for ax in [ax_vc_i12, ax_vc_i22, ax_vc_i32, ax_vc_i42]]

    letters = ['D', 'E', 'F', 'G']
    for i, wind in enumerate(t_windows):
        ax_vc_v.axvspan(wind[0], wind[1], color='grey', alpha=.5)
        ax_vc_v.text((wind[0]+wind[1])/2-20, 25, letters[i], fontsize=8)


    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, ax in enumerate(
            [ax_spont1, ax_spont2, ax_vc_v, ax_vc_v1, ax_vc_v2, ax_vc_v3, ax_vc_v4]):
        if i == 2:
            ax.set_title(letters[i], y=.98, x=-.05)
        else:
            ax.set_title(letters[i], y=.98, x=-.1)

    ax_spont1.legend()

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-drug_ap_vc_comparison.pdf', transparent=True)

    plt.show()


def plot_cc_drug(ax, f_name, t_length=9000):
    all_files = listdir('./data/cells')
    t = np.linspace(0, t_length/10, t_length)

    ap_pre = pd.read_csv(f'./data/cells/{f_name}/Pre-drug_spont.csv')
    ap_dat, cc_shape = get_ap(ap_pre, t_length)
    ax.plot(t, ap_dat, 'k', alpha = .8, label='Baseline')

    ap_post= pd.read_csv(f'./data/cells/{f_name}/Post-drug_spont.csv')
    ap_dat, cc_shape = get_ap(ap_post, t_length)
    ax.plot(t, ap_dat, 'r', alpha = .8, label='Drug')

    ax.set_ylim(-65, 30)


def plot_v(ax, t_window, is_time_sub=False):
    vc_dat = pd.read_csv(f'./data/cells/3_021021_2_alex_control/Pre-drug_vcp_70_70.csv')

    if is_time_sub:
        t_window = [t+4000 for t in t_window]
        times = vc_dat['Time (s)'][
                        t_window[0]*10:t_window[1]*10]*1000 - 4000
    else:
        times = vc_dat['Time (s)'][
                        t_window[0]*10:t_window[1]*10]*1000 - t_window[0]

    voltage = vc_dat['Voltage (V)'][t_window[0]*10:t_window[1]*10]

    ax.plot(times, voltage*1000, 'k')


def plot_i(ax, f_name, t_window, is_kernel=True, is_time_sub=True, ls='-'):
    vc_pre = pd.read_csv(f'./data/cells/{f_name}/Pre-drug_vcp_70_70.csv')

    if is_time_sub:
        t_window = [t+4000 for t in t_window]
        times = vc_pre['Time (s)'][
                        t_window[0]*10:t_window[1]*10]*1000 - 4000
    else:
        times = vc_pre['Time (s)'][
                        t_window[0]*10:t_window[1]*10]*1000 - t_window[0]

    curr = vc_pre['Current (pA/pF)'][t_window[0]*10:t_window[1]*10]
    if is_kernel:
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        curr = np.convolve(curr, kernel, mode='same')
    ax.plot(times, curr, 'k', alpha = .8, label='No Drug', linestyle=ls)

    vc_post = pd.read_csv(f'./data/cells/{f_name}/Post-drug_vcp_70_70.csv')
    curr = vc_post['Current (pA/pF)'][t_window[0]*10:t_window[1]*10]
    if is_kernel:
        curr = np.convolve(curr, kernel, mode='same')
    ax.plot(times, curr, 'r', alpha = .8, label='Drug', linestyle=ls)


def plot_vc(ax_vc_v, ax_vc_i, windows, zoom_axes):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 250, 2500)

    n = 0

    st = 40000
    end = 140000

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        exp_dat = pd.read_csv('./data/ap_features.csv')

        t = vc_dat['Time (s)'][st:end]*1000-4000

        if np.isnan(exp_dat[exp_dat['File'] == f]['CL'].values[0]):
            #FLAT
            col = '#ff7f00'
        else:
            #SPONT
            col = '#377eb8'

        ax_vc_i.plot(t, vc_dat['Current (pA/pF)'][st:end], color=col, alpha=.1)
        
        for i, w in enumerate(windows):
            z_st = st+w[0]*10
            z_end = st+w[1]*10
            t_z = vc_dat['Time (s)'][z_st:z_end]*1000-4000
            c_z = vc_dat['Current (pA/pF)'][z_st:z_end]
            zoom_axes[i][1].plot(t_z, c_z, color=col, alpha=.1)


    ax_vc_v.plot(t, vc_dat['Voltage (V)'][st:end]*1000, 'k')
    for i, w in enumerate(windows):
        z_st = st+w[0]*10
        z_end = st+w[1]*10
        t_z = vc_dat['Time (s)'][z_st:z_end]*1000-4000
        v_z = vc_dat['Voltage (V)'][z_st:z_end]*1000

        zoom_axes[i][0].plot(t_z, v_z, 'k')

    letters = ['D', 'E']

    for i, w in enumerate(windows):
        ax_vc_v.axvspan(w[0], w[1], color='#999999', alpha=.3)
        zoom_axes[i][0].axvspan(w[0], w[1], color='#999999', alpha=.2)
        ax_vc_v.text((w[0]+w[1])/2-15, 25, letters[i], fontsize=8)

    ax_vc_v.tick_params(labelbottom=False)
    ax_vc_i.set_ylim(-15, 15)

    ax_vc_v.set_ylabel('Voltage (mV)')
    ax_vc_i.set_ylabel('Current (pA/pF)')
    ax_vc_i.set_xlabel('Time (ms)')

    zoom_axes[0][1].set_ylim(-15, 15)
    zoom_axes[1][1].set_ylim(-20, 0)

    zoom_axes[0][0].set_ylabel('Voltage (mV)')
    zoom_axes[0][1].set_ylabel('Current (pA/pF)')

    zoom_axes[0][1].set_xlabel('Time (ms)')
    zoom_axes[1][1].set_xlabel('Time (ms)')


def get_ap(ap_dat, t_length=6000):
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
        #import pdb
        #pdb.set_trace()
        #return 

    idx_start = peak_idxs[0] - 100
    idx_end = idx_start + t_length
    #min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    #min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    #search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    #amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    #v_90 = min_v + amplitude * .1
    #idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    return v[idx_start:idx_end], 'spont'




def main():
    #plot_figure()
    plot_figure_big()



if __name__ == '__main__':
    main()


