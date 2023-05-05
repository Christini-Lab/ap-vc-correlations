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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import umap

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_figure():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5))
    fig.subplots_adjust(.1, .2, .95, .9, wspace=.5)

    correlation_times = [600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]# + [101*(i+1) for i in range(0, 80)]
    #correlation_times = [10*(i+1) for i in range(0, 900)]

    features = ['MP', 'APD90', 'dVdt']
    ap_features, vc_features = get_ap_vc_features(correlation_times)

    plot_pca(axs[0], 'MP', ap_features, vc_features)
    plot_pca(axs[1], 'APD90', ap_features, vc_features)
    plot_pca(axs[2], 'dVdt', ap_features, vc_features)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].legend(loc='upper right')
    #axs[0].set_xlim(-3.5, 8)
    curr_lims = axs[0].get_ylim()
    axs[0].set_ylim(curr_lims[0], curr_lims[1]*1.6)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-exp_pca.pdf', transparent=True)

    plt.show()


def get_ap_vc_features(correlation_times):
    all_ap_features = pd.read_csv('./data/ap_features.csv')
    t_window = [4000, 13500]

    all_currs_dict = {}
    all_currs_list = []

    all_files = listdir('./data/cells')
    start_idx, end_idx = t_window[0] * 10, t_window[1] * 10

    all_vc_dat = []

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        cell_params = pd.read_excel(f'./data/cells/{f}/cell-params.xlsx')

        vc_curr = vc_dat['Current (pA/pF)'][start_idx:end_idx].values
        all_vc_dat.append([f] + [vc_curr[time*10] for time in correlation_times])
        print(f)


    cols = ['File'] + [i for i in range(0, len(all_vc_dat[-1])-1)]
    all_currs_df = pd.DataFrame(all_vc_dat, columns=cols)
    all_currs_df = all_currs_df.sort_values('File')
    all_ap_features = all_ap_features.sort_values('File')
    
    return all_ap_features, all_currs_df


def plot_pca(ax, feature, ap_features, vc_features):
    sc = StandardScaler()

    X_train = sc.fit_transform(vc_features.iloc[:, 1:].values)

    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(X_train)
    #X_train = TSNE(n_components=2).fit_transform(X_train)
    #reducer = umap.UMAP(n_components=2).fit_transform(X_train)
    norm_ap = NormalizeData(ap_features[feature].values)

    mask = ~np.isnan(ap_features[feature].values)
    curr_features = ap_features[feature].values[mask]
    #TODO: Figure this out
    if feature == 'APD90':
        curr_features[curr_features==curr_features.max()]=200
    if feature == 'dVdt':
        curr_features[curr_features>50] = 50
        
    X_train = X_train[mask, :]
    norm_ap = NormalizeData(curr_features)

    colors = np.array([[v,v,0] for v in norm_ap])
    
    #ap_features[feature].values[np.isnan(
    #       ap_features[feature].values)] = (ap_features[feature].min())

    #curr_features = ap_features[feature]

    #colors = []
    #for i in range(0, X_train.shape[0]):
    #    if mask[i] == True:
    #        colors.append([1, 1, 1])
    #    else:
    #        colors.append([norm_ap[i], norm_ap[i], 0])

    #colors = np.array(colors)

    corr_vals = pearsonr(X_train[:,0], curr_features)
    ax.set_xlabel(f'PC 1 (r={round(corr_vals[0], 5)}, p={round(corr_vals[1], 5)})')
    corr_vals = pearsonr(X_train[:,1], curr_features)
    ax.set_ylabel(f'PC 2 (r={round(corr_vals[0], 5)}, p={round(corr_vals[1], 5)})')

    if feature == 'MP':
        is_ap_mask = ~np.isnan(ap_features['APD90'].values)
        curr_pca_dat = X_train[is_ap_mask, :]
        curr_colors_dat = colors[is_ap_mask, :]
        [ax.scatter(curr_pca_dat[i, 0], curr_pca_dat[i, 1], c=curr_colors_dat[i, :], marker='o') for i in range(0, curr_pca_dat.shape[0])]
        i = 0
        ax.scatter(curr_pca_dat[i, 0], curr_pca_dat[i, 1], c=curr_colors_dat[i, :], marker='o', label='Has AP')
        
        is_ap_mask = np.isnan(ap_features['APD90'].values)
        curr_pca_dat = X_train[is_ap_mask, :]
        curr_colors_dat = colors[is_ap_mask, :]
        [ax.scatter(curr_pca_dat[i, 0], curr_pca_dat[i, 1], c=curr_colors_dat[i, :], marker='^') for i in range(0, curr_pca_dat.shape[0])]
        i = 0
        ax.scatter(curr_pca_dat[i, 0], curr_pca_dat[i, 1], c=curr_colors_dat[i, :], marker='^', label='No AP')

    else:
        [ax.scatter(X_train[i, 0], X_train[i, 1], c=colors[i, :]) for i in range(0, X_train.shape[0])]
    ax.set_title(feature)



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))




def plot_i_ap_corr(corr_axs, feature, correlation_times, heatmap_ax=None):
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

    seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    feature_cols = {'MP': '#4daf4a', 'APD90': 'purple', 'dVdt': 'orange'}
    feature_names = {'MP': 'MP (mV)', 'APD90': r'$APD_{90}$ (ms)', 'dVdt': r'$dV/dt_{max}$ (V/s)'}

    for i, corr_time in enumerate(correlation_times):
        i_curr = valid_vc_dat[:, int(corr_time*10)]

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
        heatmap(corr, ax=heatmap_ax, annot=True, annot_kws={"size":6}, mask=np.triu(corr), xticklabels=seg_names,yticklabels=seg_names)#, cbar=False)

    corr_axs[3].set_ylabel(feature_names[feature])


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]
    return np.array(new_vals)


def main():
    plot_figure()


if __name__ == '__main__':
    main()
