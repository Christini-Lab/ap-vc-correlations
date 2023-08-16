from os import listdir
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from sklearn import linear_model

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
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4))
    fig.subplots_adjust(.1, .11, .95, .95, hspace=.3)

    directory = 'pop_7_Paci'
    plot_v(axs[0], directory)

    t_window = [50000, 59500]
    #directory = 'pop_5_Paci'
    #plot_mod_dat(directory, axs, t_window)  

    plot_mod_dat(directory, axs, t_window)  

    t_window = [4000, 13500]
    plot_exp_dat(axs, t_window)

    letters = ['A', 'B', 'C', 'D', 'E']
    ylabs = ['Voltage (mV)', 'Current (pA/pF)', 'r of MP', r'r of $APD_{90}$', 'r of dV/dt']
    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(letters[i], y=.98, x=-.08)
        ax.set_ylabel(ylabs[i])

    axs[1].legend()
    axs[1].set_ylim(-15, 18)
    #axs[0].set_xlim(400, 900)

    axs[-1].set_xlabel('Time (ms)')

    [ax.set_ylim(-1, 1) for ax in axs[2:]]

    if 'Kernik' in directory:
        plt.savefig('./figure-pdfs/f-mod_exp_corr_comp_kernik.pdf', transparent=True)
    else:
        plt.savefig('./figure-pdfs/f-mod_exp_corr_comp_paci.pdf', transparent=True)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-mod_exp_comp.pdf', transparent=True)

    plt.show()


def plot_v(ax, directory):
    t = np.loadtxt(f'./data/mod_populations/{directory}/vc_times.csv') -50000
    v = np.loadtxt(f'./data/mod_populations/{directory}/vc_v.csv')

    ax.plot(t, v, 'k')
    ax.set_ylabel('Voltage (mV)')


def plot_exp_dat(axs, t_window):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_ap_features = pd.read_csv('./data/ap_features.csv')

    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10

    all_currs_dict = {}
    all_currs_list = []

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

    all_currs_array = np.array(all_currs_list)
    mean_curr = all_currs_array.mean(0)
    max_curr = all_currs_array.max(0)
    min_curr = all_currs_array.min(0)

    [axs[1].plot(times, c, 'k', alpha=.05) for c in all_currs_list]

    #axs[1].plot(times, mean_curr, 'k', label=r'Mean $I_{out}$')
    #axs[1].fill_between(times, min_curr, max_curr, edgecolor=None, facecolor='k', alpha=.1)#, label=r'Range of  $I_{out}$')
    return

    # Make all_ap_features rows and all_currs_df in same order
    #all_currs_df = pd.DataFrame(all_currs_dict)
    #all_currs_df = all_currs_df.reindex(
    #        sorted(all_currs_df.columns), axis=1)
    #all_ap_features = all_ap_features.sort_values('File')

    #cols = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c'] 

    #for j, feature in enumerate(['MP', 'APD90', 'dVdt']):
    #    valid_indices = np.invert(np.isnan(all_ap_features[feature]))
    #    
    #    valid_ap_dat= all_ap_features[valid_indices]
    #                
    #    valid_vc_dat = np.array([all_currs_df[col_name] for col_name in valid_ap_dat['File']])

    #    num_vc_pts = valid_vc_dat.shape[1]

    #    feature_corrs = [np.corrcoef(valid_ap_dat[feature].values,
    #                        valid_vc_dat[:, i])[0][1] for i in
    #                                            range(0, num_vc_pts)]
    #    
    #    axs[j+2].plot(times, feature_corrs, color='k',
    #                                    label=feature, alpha=.7)
    #    axs[j+2].axhline(0, color='grey', alpha=.5, linestyle='--')

    #    corr_idx = np.argmax(np.abs(feature_corrs))

    #    #ax_v.axvline(times[corr_idx], color=cols[j], linestyle='--', linewidth=1.6)
    #    #ax_i.axvline(times[corr_idx], color=cols[j], linestyle='--', linewidth=1.6)
    #    #ax_corr.axvline(times[argmax_feature], color=cols[j], linestyle='--')


def plot_mod_dat(directory, axs, t_window):
    path = f'./data/mod_populations/{directory}'

    vc_t = np.loadtxt(f'{path}/vc_times.csv') - 50000
    vc_iout = np.load(f'{path}/vc_iout.npy')

    mean_curr = vc_iout.mean(0)
    max_curr = vc_iout.max(0)
    min_curr = vc_iout.min(0)

    if 'Paci' in directory:
        col = '#E66100'
        name = 'Paci'
        sty = '--'
    else:
        col = '#5D3A9B'
        name = 'Kernik'
        sty = 'dotted'

    axs[1].plot(vc_t, mean_curr, color=col, linestyle=sty, label=name)
    axs[1].fill_between(vc_t, min_curr, max_curr, edgecolor=None, facecolor=col, alpha=.15)#, label=r'Range of  $I_{out}$')

    #ax_i.set_ylabel('Current (pA/pF)')
    #ax_i.set_ylim(-15, 15)

    #CORRELATIONS
    #cols = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']

    #ap_features = pd.read_csv(f'{path}/all_ap_features.csv')

    #for j, feature in enumerate(['mp', 'apd90', 'dvdt']):
    #    valid_indices = np.invert(np.isnan(ap_features[feature]))

    #    valid_ap_features = ap_features[valid_indices]
    #    valid_vc_dat = vc_iout[valid_indices, :]

    #    num_vc_pts = valid_vc_dat.shape[1]

    #    import time
    #    st_t = time.time()
    #    print('start time')
    #    feature_corrs = [np.corrcoef(valid_ap_features[feature].values,
    #                        valid_vc_dat[:, i])[0][1] for i in
    #                                            range(0, num_vc_pts)]
    #    print(f'It took {time.time() - st_t}')

    #    axs[j+2].plot(vc_t, feature_corrs, color=col, linestyle=sty,
    #                            label=feature, linewidth=1.5)
    #    #axs[j+2].label(feature)

    #    #corr_idx = np.argmax(np.abs(feature_corrs))

    #    #ax_v.axvline(vc_t[corr_idx], color=cols[j],
    #    #                    linestyle='--', linewidth=1.6)
    #    #ax_i.axvline(vc_t[corr_idx], color=cols[j],
    #    #                    linestyle='--', linewidth=1.6)

    #    #x = valid_vc_dat[:, corr_idx]
    #    #y = valid_ap_features[feature].values

    #    #regplot(x=x, y=y, ax=corr_zoom_axes[j], color=cols[j])
    #    #corr_zoom_axes[j].set_xlabel(r'$I_{out}$ '+ f'at t={int(vc_t[corr_idx])} (pA/pF)')
    #    #corr_zoom_axes[j].set_ylabel(feature)


    #    print(f'finished {feature}')


    #ax_corr.set_ylabel('r')
    #ax_corr.legend(framealpha=1)
    #ax_corr.set_xlabel('Time (ms)')

    #ax_corr.set_ylim(-1, 1)


def plot_i_corr(ax_v, ax_i, ax_corr, corr_zoom_axes,
                                    t_window, mv_avg, directory):

    path = f'./data/mod_populations/{directory}'

    vc_t = np.loadtxt(f'{path}/vc_times.csv') - 50000
    vc_iout = np.load(f'{path}/vc_iout.npy')

    cc_v = np.load(f'{path}/cc_v.npy')

    for i in range(0, len(vc_iout)):
        peak_pts = find_peaks(cc_v[i], 10, distance=100, width=20)[0]
        if len(peak_pts) > 0:
            ax_i.plot(vc_t, vc_iout[i])

    ax_i.set_ylabel('Current (pA/pF)')
    ax_i.set_ylim(-45, 25)

    #CORRELATIONS
    cols = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']

    ap_features = pd.read_csv(f'{path}/all_ap_features.csv')

    for j, feature in enumerate(['mp', 'apd90', 'dvdt']):
        valid_indices = np.invert(np.isnan(ap_features[feature]))

        valid_ap_features = ap_features[valid_indices]
        valid_vc_dat = vc_iout[valid_indices, :]

        num_vc_pts = valid_vc_dat.shape[1]

        import time
        st_t = time.time()
        print('start time')
        feature_corrs = [np.corrcoef(valid_ap_features[feature].values,
                            valid_vc_dat[:, i])[0][1] for i in
                                                range(0, num_vc_pts)]
        print(f'It took {time.time() - st_t}')

        ax_corr.plot(vc_t, feature_corrs, color=cols[j],
                                        label=feature, alpha=.7, linewidth=1.6)

        corr_idx = np.argmax(np.abs(feature_corrs))
        #if feature == 'mp':
        #    corr_idx = 35430 
        #if feature == 'dvdt':
        #    corr_idx = 12580


        ax_v.axvline(vc_t[corr_idx], color=cols[j],
                            linestyle='--', linewidth=1.6)
        ax_i.axvline(vc_t[corr_idx], color=cols[j],
                            linestyle='--', linewidth=1.6)

        x = valid_vc_dat[:, corr_idx]
        y = valid_ap_features[feature].values

        regplot(x=x, y=y, ax=corr_zoom_axes[j], color=cols[j])
        corr_zoom_axes[j].set_xlabel(r'$I_{out}$ '+ f'at t={int(vc_t[corr_idx])} (pA/pF)')
        corr_zoom_axes[j].set_ylabel(feature)


        print(f'finished {feature}')


    ax_corr.set_ylabel('r')
    ax_corr.legend(framealpha=1)
    ax_corr.set_xlabel('Time (ms)')

    ax_corr.set_ylim(-1, 1)


def main():
    plot_figure()


if __name__ == '__main__':
    main()

