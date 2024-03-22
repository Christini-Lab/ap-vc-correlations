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


def plot_figure_only_ap():
    fig = plt.figure(figsize=(4.5, 4))
    fig.subplots_adjust(.15, .12, .95, .95)

    grid = fig.add_gridspec(1, 1)

    ax_spont = fig.add_subplot(grid[0])

    plot_cc(ax_spont, 'flat', 'k')
    plot_cc(ax_spont, 'spont', 'k')

    ax_spont.spines['top'].set_visible(False)
    ax_spont.spines['right'].set_visible(False)

    ax_spont.set_ylabel('Voltage (mV)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-ap_my_dat_hetero.pdf', transparent=True)

    plt.show()


def plot_cc(ax, cc_type, col=None):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 250, 2500)

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

            #if f == '6_033021_4_alex_control':
            #    ax.plot(t, ap_dat, 'pink', rasterized=True)
            #if f == '4_022421_1_alex_cisapride':
            #    ax.plot(t, ap_dat, 'pink', rasterized=True)


            #ax.plot(ap_dat, 'k', alpha = .3)
            n += 1
        else:
            continue

    #if cc_type == 'spont':
    #    ax.plot(t_sp, ap_dat_sp, 'pink')

    #ax.text(100, 40, f'n={n}')
    ax.set_ylim(-70, 50)
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
        #import pdb
        #pdb.set_trace()
        #return 

    idx_start = peak_idxs[0] - 50
    idx_end = idx_start + 2500
    #min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    #min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    #search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    #amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    #v_90 = min_v + amplitude * .1
    #idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    return v[idx_start:idx_end], 'spont'


#UTILITY
def main():
    plot_figure_only_ap()


if __name__ == '__main__':
    main()


