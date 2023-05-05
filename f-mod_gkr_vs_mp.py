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
import myokit


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
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.5))
    fig.subplots_adjust(.1, .15, .95, .95, wspace=.25)

    plot_kernik_ikr_knockdown(axs)

    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('MDP (mV)')
    axs[0].set_xlabel('Time (ms)')
    axs[1].set_xlabel(r'Scaled $g_{Kr}$')

    letters = ['A', 'B', '', 'C']

    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(letters[i], y=.94, x=-.2)

    axs[1].text(1, -33, 'Exp 90%')
    axs[1].text(1, -65, 'Exp 10%')

    axs[0].arrow(250, -75, 0, 75, head_width=20, head_length=4, facecolor='rred', color='red')
    axs[0].text(200, -80, r'More $I_{Kr}$', color='red')
    axs[0].text(200, 5, r'Less $I_{Kr}$', color='red')
    axs[0].set_ylim(-83, 20)

    plt.savefig('./figure-pdfs/f-gkr-vs-mdp.pdf')#, transparent=True)
    plt.show()


def plot_kernik_ikr_knockdown(axs):
    g_scales = 10**np.linspace(-1, 0.2, 10)

    for g_scale in g_scales:
        mod, p, x = myokit.load('./mmt/kernik_leak_fixed.mmt')
        #mod, p, x = myokit.load('./mmt/paci_leak_ms_fixed.mmt')
        mod['ikr']['g_Kr'].set_rhs(g_scale) #Kernik
        #mod['ikr']['g'].set_rhs(g_scale) #Paci

        sim = myokit.Simulation(mod)
        prepace = 100000
        sim.pre(prepace)

        dat = sim.run(10000, log_times=np.arange(0, 10000, 1))

        t = dat.time()
        v = dat['membrane.V']

        t, v = get_ap(t, v)

        axs[0].plot(t, v, 'k')
        axs[1].scatter(g_scale, np.min(v), color='k')
        print(g_scale)
    
    axs[1].axhspan(-65, -34, color='yellow', alpha=.15)
    axs[0].set_xlim(-100, 400)


def get_ap(t, v):
    if ((np.max(v) - np.min(v)) < 40):
        t = np.array(t) - 9600
        return t[-500:], v[-500:]

    peak_idxs = find_peaks(np.diff(v), height=.1, distance=100)[0]
    st_idx = peak_idxs[-2] - 100
    end_idx = st_idx + 500 

    t = np.array(t)-t[peak_idxs[-2]]

    return t[st_idx:end_idx], v[st_idx:end_idx]

plot_figure()
