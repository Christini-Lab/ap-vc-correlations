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


def plot_figure(feature, heat_currents):
    fig = plt.figure(figsize=(3.7, 9))
    fig.subplots_adjust(.17, .05, .95, .98)

    grid = fig.add_gridspec(9, 3, hspace=1, wspace=1.4)

    sub = grid[0:3, :]
    corr_subgrid = sub.subgridspec(3, 3, hspace=.3)
    corr_axs = []

    for row in range(0, 3):
        for col in range(0, 3):
            ax = fig.add_subplot(corr_subgrid[row, col])
            if col != 0:
                ax.set_yticklabels('')

            corr_axs.append(ax)

    corr_axs[0].set_xticks([-150, -50])
    corr_axs[1].set_xlim(-1, 12)
    corr_axs[6].set_xlim(-11, -2)

    #heatmap_ax = fig.add_subplot(grid[0:3, 3:])

    plot_i_ap_corr(corr_axs, feature, None, heat_currents)
    corr_axs[-2].set_xlabel(r'$I_{out}$ (pA/pF)')

    all_ap_ax = fig.add_subplot(grid[3:6, :])
    plot_ap(all_ap_ax)

    sub = grid[6:, :]
    vc_subgrid = sub.subgridspec(2, 1, hspace=.4, wspace=.4, height_ratios=(1, 2))

    v_axs = []
    i_axs = []

    for row in range(0, 2):
        for col in range(0, 1):
            ax = fig.add_subplot(vc_subgrid[row, col])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0:
                v_axs.append(ax)
            else:
                i_axs.append(ax)

    num = 0
    corr_name = 'IKr'
    plot_vc(v_axs[num], i_axs[num], corr_name, xrange=[-15, 15])
    i_axs[num].set_ylim(-5, 7)
    i_axs[num].set_title(r'$I_{Kr}$', y=.9)

    #num = 1
    #corr_name = 'Ito'
    #plot_vc(v_axs[num], i_axs[num], corr_name, xrange=[-20, 20])
    #i_axs[num].set_ylim(-2, 16)
    #i_axs[num].set_title(r'$I_{to}$', y=.9)

    #num = 2
    #corr_name = 'IK1'
    #plot_vc(v_axs[num], i_axs[num], corr_name, xrange=[-100, 25])
    #i_axs[num].set_ylim(-15, 5)
    #i_axs[num].set_title(r'$I_{K1}$', y=.9)

    #num = 3
    #corr_name = 'If'
    #plot_vc(v_axs[num], i_axs[num], corr_name, xrange=[-800, 50])
    #i_axs[num].set_ylim(-18, 2)
    #i_axs[num].set_title(r'$I_{f}$', y=.9)

    #num = 4
    #corr_name = 'IKs'
    #plot_vc(v_axs[num], i_axs[num], corr_name, xrange=[-300, 40])
    #i_axs[num].set_ylim(-2, 15)
    #i_axs[num].set_title(r'$I_{Ks}$', y=.9)

    v_axs[0].set_ylabel('Voltage (mV)')
    i_axs[0].set_ylabel(r'$I_{out}$ (A/F)')

    for ax in i_axs:
        ax.set_xlabel('Time (ms)')

    letters = ['A', 'B', 'C']
    #for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v, ax_vc_v1, ax_vc_v2]):
    for i, ax in enumerate([corr_axs[0], all_ap_ax, v_axs[0]]):
        ax.set_title(letters[i], y=.97, x=-.1)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-mp-vc.pdf', transparent=True)

    plt.show()


def plot_i_ap_corr(corr_axs, feature, heatmap_ax=None, heat_currents=None):
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

    current_indices = {'I6mV':0, 'IKr':1, 'ICaL':2, 'INa2':3, 'Ito':4, 'IK1':5, 'If':6, 'IKs':7}

    seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na2}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    correlation_times = [600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]
    seg_type = ['avg', 'avg', 'min', 'min', 'max', 'avg', 'avg', 'avg']

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
            regplot(x=i_curr, y=valid_ap_dat[feature].values, ax=corr_axs[i], color='k', ci=None)


            circled_files = ['3_021121_1_alex_cisapride',
                            '6_040621_3_alex_quinidine']

            indices = [i for i, row in enumerate(valid_ap_dat.values) if row[0] in circled_files]

            corr_axs[i].scatter(i_curr[indices[0]],
                    valid_ap_dat['MP'].values[indices[0]],
                    s=40, marker='s', color='#377eb8')
                    #facecolor='none', linewidth=1.5)
            #corr_axs[i].scatter(i_curr[indices[2]],
            #                    valid_ap_dat['APD90'].values[indices[2]],
            #                    s=60, marker='s', edgecolor='#ff7f00',
            #                    facecolor='none')
            corr_axs[i].scatter(i_curr[indices[1]],
                                valid_ap_dat['MP'].values[indices[1]],
                                s=60, marker='^', color='#4daf4a')
                                #facecolor='none', linewidth=1.5)

        else:
            corr_axs[i].set_title(f'{seg_names[i]}', fontsize=8, y=.77)
            if i == 0:
                corr_axs[i].text(-110, -25, f'{seg_names[i]}', fontsize=8)

            regplot(x=i_curr, y=valid_ap_dat[feature].values, ax=corr_axs[i], color='grey', ci=None)
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

        heatmap(corr, ax=heatmap_ax, annot=True, annot_kws={"size":8}, mask=np.triu(corr), xticklabels=seg_names,yticklabels=seg_names, fmt=txt_fmt, cbar=False)

    corr_axs[3].set_ylabel(feature_names[feature])


def plot_ap(ax):
    plot_cc(ax, 'spont', 'k')
    plot_cc(ax, 'flat', 'k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Voltage (mV)')


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
                    ax.plot(t, ap_dat, col, alpha = .06, rasterized=True)
                t_sp = t
                ap_dat_sp = ap_dat
            else:
                if col is None:
                    ax.plot(t, ap_dat, '#ff7f00', alpha = .5, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = .06, rasterized=True)

            #if f == '4_021921_1_alex_control':
            #    ax.plot(t, ap_dat, 'red', rasterized=True)
            #if f == '3_021121_1_alex_cisapride':
            #    ax.plot(t, ap_dat, 'red', rasterized=True)
            if f == '3_021121_1_alex_cisapride':
                #ax.plot(t, ap_dat, '#377eb8', rasterized=True)
                t_hyper, ap_hyper = t, ap_dat
            #if f == '6_040921_1_alex_quinidine':
                #ax.plot(t, ap_dat, '#ff7f00', rasterized=True)
            #    t_med, ap_med = t, ap_dat
            if f == '6_040621_3_alex_quinidine':
                #ax.plot(t, ap_dat, '#ff7f00', rasterized=True)
                t_depol, ap_depol= t, ap_dat
            n += 1
        else:
            continue

    try:
        ax.plot(t_hyper, ap_hyper, '#377eb8', rasterized=True, label='Cell 1', linewidth=1.5)
    except:
        pass

    try:
        ax.plot(t_depol, ap_depol, '#4daf4a', rasterized=True, label='Cell 2', linewidth=1.5)
    except:
        pass

    #ax.scatter(40, -40, s=60, marker='s', color='#377eb8')
    #ax.scatter(100, 10, s=60, marker='^', color='#4daf4a')

    #if cc_type == 'spont':
    #    ax.plot(t_sp, ap_dat_sp, 'pink')

    #ax.text(100, 40, f'n={n}')
    ax.set_ylim(-75, 50)
    ax.set_xlabel('Time (ms)')
    ax.legend()


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
    idx_end = idx_start + 2500

    return v[idx_start:idx_end], 'spont'


def plot_vc(v_ax, i_ax, corr_name, xrange):
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    t_window = [4000, 13500]

    i_names = ['INa1', 'I6mV', 'IKr', 'ICaL', 'INa2', 'Ito', 'IK1', 'If', 'IKs']
    correlation_times = [501.5, 600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]

    corr_time_dict = dict(zip(i_names, correlation_times))
    corr_time = corr_time_dict[corr_name]

    all_currs_list = []

    all_files = listdir('./data/cells')
    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10
    st, end = corr_time*10 + xrange[0]*10, corr_time*10 + xrange[1]*10

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        #cell_params = pd.read_excel(f'./data/cells/{f}/cell-params.xlsx')

        vc_curr = vc_dat['Current (pA/pF)'][start_idx:end_idx].values
        vc_curr = vc_curr[st:end]
        times = vc_dat['Time (s)'][start_idx:end_idx]*1000 - t_window[0]
        times = times.values[st:end]

        i_ax.plot(times, vc_curr, color='k', alpha=.1)

        if f == '3_021121_1_alex_cisapride':
            vc_sh = vc_curr 
        if f == '6_040621_3_alex_quinidine':
            vc_long = vc_curr 

    i_ax.plot(times, vc_sh, color='#377eb8', linewidth=1.5)
    i_ax.plot(times, vc_long, color='#4daf4a', linewidth=1.5)

    v = vc_dat['Voltage (V)'][start_idx:end_idx].values
    v = v[st:end] * 1000

    v_ax.plot(times, v, color='k')
    v_ax.axvline(corr_time, color='red', alpha=.6, linestyle='--')
    i_ax.axvline(corr_time, color='red', alpha=.6, linestyle='--')


def plot_gkr_vs_mp():
    directory = 'pop_8_Kernik'
    all_ind_params = pd.read_csv(f'./data/mod_populations/{directory}/all_params.csv')
    all_ap_features = pd.read_csv(f'./data/mod_populations/{directory}/all_ap_features.csv')
    all_ap_features = all_ap_features.iloc[:75, :].copy()
    all_ind_params = all_ind_params.iloc[:75, :].copy()
    
    plt.scatter(all_ind_params['ikr.g_Kr'], all_ap_features['mp'])
    plt.show()
    import pdb
    pdb.set_trace()
    

def plot_kernik_ikr_knockdown():

    fig, ax = plt.subplots(1,1)

    for g_scale in np.arange(.1, 2, .1):
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

        ax.scatter(g_scale, np.min(v), color='k')
        print(g_scale)

    
    ax.axhspan(-65, -34, color='yellow', alpha=.3)
    plt.show()
    import pdb
    pdb.set_trace()






def main():
    plot_figure('MP', heat_currents=None)
    #plot_gkr_vs_mp()
    #plot_kernik_ikr_knockdown()


if __name__ == '__main__':
    main()

