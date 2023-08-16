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
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 4))

    high_var_cells = [
            #'6_033021_4_alex_control',
            #'4_022621_4_alex_cisapride',
            #'4_022421_1_alex_cisapride',
            #'4_021921_2_alex_control',
            #'6_033021_5_alex_control'
            #'7_042621_2_alex_quinine',
            '4_022621_2_alex_cisapride',
            '7_042721_2_alex_quinine'] 

    plot_aps(axs[0], files=high_var_cells, cols=[(1, 0, 0), (0, .6, 0)])
    axs[0].set_title('High CL Variance')

    low_var_cells = [#'7_042721_4_alex_quinine',
                     #'4_022521_2_alex_cisapride',
                     '7_042621_6_alex_quinine', 
                     '4_021921_1_alex_control']

    plot_aps(axs[1], files=low_var_cells, cols=[(0, 0, 1), (1, .65, 0)])
    axs[1].set_title('Low CL Variance')

    matplotlib.rcParams['pdf.fonttype'] = 42

    plt.savefig('./figure-pdfs/f-high-low-variance.pdf')#, transparent=True)


    plt.show()


def plot_aps(ax, files, cols):
    end_time = 3200 
    t = np.linspace(0, end_time, end_time*10)

    for i, f in enumerate(files):
        ap_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')
        ap_dat, cc_shape = get_ap(ap_dat, end_time*10)
        #ax.plot(t, ap_dat, '#377eb8', alpha = .3, rasterized=True)
        ax.plot(t, ap_dat, c=cols[i], alpha = .7, rasterized=True)

    ax.set_ylim(-70, 42)
    ax.set_xlabel('Time (ms)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Voltage (mV)')


def plot_ap(ax):
    plot_cc(ax, 'spont', 'k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Voltage (mV)')


def get_ap(ap_dat, t_amount):
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
    idx_end = idx_start + t_amount 

    return v[idx_start:idx_end], 'spont'


def main():
    plot_figure()


if __name__ == '__main__': 
    main()
